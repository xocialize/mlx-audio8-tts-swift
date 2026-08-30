// audio8-gates — Stage 1 parity gates as CLI modes (swift run audio8-gates <mode>).
// Fixtures: oracle-capture/goldens/arktts_goldens.safetensors (fp32 PT oracle dumps;
// codec tensors channels-FIRST — transposed here at the gate site).
//
// Modes:
//   --s0   key contract (LM module keys == model.safetensors keys; codec key probe)
//   --s1   LM + codec unit gates vs oracle fixtures (CPU stream, fp32)
//   --s2   prefill logits, codec encode code-exactness, greedy generation token-exactness
//   --s2b  prompt ids vs processor fixtures + one real sampled GPU generation (wav out)

import Foundation
@_spi(FalconH1Encoder) import MLXLLM
import MLXLMCommon
import MLX
import Audio8TTSCore
import MLXAudio8TTS
import MLXToolKit

// MARK: paths

let fileDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let repoRoot = fileDir.deletingLastPathComponent().deletingLastPathComponent()
let workspaceRoot = repoRoot.deletingLastPathComponent()
// Both overridable so the same gates run against either checkpoint: the 0.6b (default)
// or the 0.1b, whose goldens and release repo live under WIP/audio8-tts.
let goldensURL = ProcessInfo.processInfo.environment["AUDIO8_GOLDENS"].map { URL(fileURLWithPath: $0) }
    ?? workspaceRoot.appendingPathComponent("oracle-capture/goldens/arktts_goldens.safetensors")
let modelDir = ProcessInfo.processInfo.environment["AUDIO8_MODEL_DIR"].map { URL(fileURLWithPath: $0) }
    ?? workspaceRoot.appendingPathComponent("release/Audio8-TTS-Preview-0.6b-bf16")

/// The right package for whatever `AUDIO8_MODEL_DIR` points at.
///
/// The two Audio8 checkpoints are separate `ModelPackage` types because their manifests differ
/// (licence, footprints, provenance, surface). Their behaviour does not — both delegate to
/// `Audio8Runtime` — so the live gates below run either one through an existential, chosen from
/// the checkpoint's own config.json rather than from a flag nobody would remember to pass.
func makeAudio8Package(_ dir: URL) throws -> any ModelPackage & StreamEmitting {
    let config = try ArkttsConfig.load(from: dir.appendingPathComponent("config.json"))
    return config.usesFalconSlow
        ? Audio8MiniPackage(configuration: Audio8MiniConfiguration(modelDirectory: dir))
        : Audio8Package(configuration: Audio8Configuration(modelDirectory: dir))
}

func stderrPrint(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

var failures: [String] = []

func loadGoldens() -> [String: MLXArray] {
    guard let goldens = try? loadArrays(url: goldensURL) else {
        fatalError("cannot load goldens at \(goldensURL.path)")
    }
    return goldens
}

/// torch channels-first (B, C, T) -> NLC (B, T, C)
func toNLC(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 1) }

func check(_ name: String, _ got: MLXArray, _ golden: MLXArray, tol: Float) {
    eval(got)
    let gotF = got.asType(.float32)
    let goldenF = golden.asType(.float32)
    guard gotF.shape == goldenF.shape else {
        failures.append("\(name): shape \(gotF.shape) vs \(goldenF.shape)")
        print("  FAIL \(name): shape \(gotF.shape) vs \(goldenF.shape)")
        return
    }
    let diff = abs(gotF - goldenF).max().item(Float.self)
    let status = diff < tol ? "ok  " : "FAIL"
    if diff >= tol { failures.append("\(name): max_abs \(diff)") }
    print("  \(status) \(name)  max_abs \(diff)")
}

func finish(_ gate: String) {
    if failures.isEmpty {
        print("\n\(gate): ALL CHECKS PASSED")
    } else {
        print("\n\(gate): \(failures.count) FAILURES")
        failures.forEach { print(" - \($0)") }
        exit(1)
    }
}

// MARK: wav writer (16-bit PCM mono)

func writeWav(_ samples: [Float], to url: URL, sampleRate: Int = 44100) throws {
    var data = Data()
    let payload = samples.count * 2
    func append(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
    func append16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
    data.append("RIFF".data(using: .ascii)!); append(UInt32(36 + payload))
    data.append("WAVE".data(using: .ascii)!); data.append("fmt ".data(using: .ascii)!)
    append(16); append16(1); append16(1); append(UInt32(sampleRate))
    append(UInt32(sampleRate * 2)); append16(2); append16(16)
    data.append("data".data(using: .ascii)!); append(UInt32(payload))
    for sample in samples {
        let clamped = max(-1, min(1, sample))
        append16(UInt16(bitPattern: Int16(clamped * 32767)))
    }
    try data.write(to: url)
}

// MARK: gates

func gateS0() throws {
    let model = try Audio8TTS.load(directory: modelDir)
    // load() already throws on any key-contract violation; report the counts.
    let lmKeys = model.lm.parameters().flattened().map(\.0)
    print("  ok   LM key contract: \(lmKeys.count) parameters, 0 missing / 0 unused")
    // codec probe: every key the forward references must exist (encode+decode walk)
    let probe = MLXArray.zeros([1, ArkttsCodec.frameLength * 2], dtype: .float32)
    let (codes, length) = model.codec.encode(probe, sampleCount: ArkttsCodec.frameLength * 2)
    eval(codes)
    let wave = model.codec.decode(codes)
    eval(wave)
    print("  ok   codec key probe: encode -> \(codes.shape) (len \(length)), decode -> \(wave.shape)")
    finish("S0")
}

func gateS1() throws {
    Device.setDefault(device: Device(.cpu))
    let goldens = loadGoldens()
    let model = try Audio8TTS.load(directory: modelDir, lmDtype: .float32)
    let lm = model.lm

    check("unit.rope_table_fast", lm.fastFreqsCis.values, goldens["unit.rope_table_fast"]!, tol: 1e-6)

    // The slow-stack unit fixtures differ per backbone: the arktts path has a rope table and
    // hand-written blocks to probe, the falcon_h1 path has neither (its rope is internal and
    // its layers come from mlx-swift-lm). Each variant's goldens are named accordingly, so
    // running the wrong set would fail on a missing key rather than silently skip.
    let xSlow: MLXArray
    if lm.config.usesFalconSlow {
        // Layer-level, not submodule-level: the decoder layer's mamba/attention/MLP are
        // internal to mlx-swift-lm and deliberately stay that way (the exposed surface
        // mirrors the Gemma encoder exposures). The layer input is captured from a real
        // prefill, so this probes actual distributions rather than synthetic noise.
        let layerIn = goldens["unit.inputnorm0_in"]!
        check(
            "prefill.slow_layer0_out",
            lm.slow!.layers[0](
                layerIn, cache: nil,
                attnMask: createAttentionMask(h: layerIn, cache: nil),
                mambaMask: createSSMMask(h: layerIn, cache: nil)),
            goldens["prefill.slow_layer0_out"]!, tol: 1e-3)
        check("unit.finalnorm_out", lm.slow!.finalLayerNorm(goldens["unit.finalnorm_in"]!),
              goldens["unit.finalnorm_out"]!, tol: 1e-4)
        // The seam itself, and the trap: the multiplier belongs on the COMPOSITE.
        check("embed.raw_unscaled", lm.embed(goldens["prompt.ids"]!.asType(.int32)),
              goldens["embed.raw_unscaled"]!, tol: 1e-4)
        xSlow = goldens["unit.x_fast_in"]!
    } else {
        check("unit.rope_table_slow", lm.freqsCis!.values[..<32],
              goldens["unit.rope_table_slow"]!, tol: 1e-6)
        let x = goldens["unit.x_slow_in"]!
        let length = 17
        let rope = lm.freqsCis!.values[MLXArray(0..<Int32(length))].expandedDimensions(axis: 0)
        let row = MLXArray(0..<Int32(length))[0..., .newAxis]
        let col = MLXArray(0..<Int32(length))[.newAxis, 0...]
        let mask = (col .<= row).expandedDimensions(axes: [0, 1])
        check("unit.rmsnorm_out", lm.norm!(x), goldens["unit.rmsnorm_out"]!, tol: 1e-5)
        check("unit.slow_ffn0_out", lm.layers![0].feedForward(x),
              goldens["unit.slow_ffn0_out"]!, tol: 1e-4)
        check(
            "unit.slow_attn0_out",
            lm.layers![0].attention(lm.layers![0].attentionNorm(x), rope: rope, mask: mask),
            goldens["unit.slow_attn0_out"]!, tol: 1e-4)
        check("unit.slow_block0_out", lm.layers![0](x, rope: rope, mask: mask),
              goldens["unit.slow_block0_out"]!, tol: 1e-4)
        xSlow = x
    }
    let xFast = xSlow[0..., ..<10]
    let ropeF = lm.fastFreqsCis.values[MLXArray(0..<10)].expandedDimensions(axis: 0)
    let rowF = MLXArray(0..<10)[0..., .newAxis]
    let colF = MLXArray(0..<10)[.newAxis, 0...]
    let maskF = (colF .<= rowF).expandedDimensions(axes: [0, 1])
    check("unit.fast_block0_out", lm.fastLayers[0](xFast, rope: ropeF, mask: maskF),
          goldens["unit.fast_block0_out"]!, tol: 1e-4)

    let codec = model.codec
    check("unit.codec_encoder_out", codec.encoderForward(toNLC(goldens["unit.x_audio_in"]!)),
          toNLC(goldens["unit.codec_encoder_out"]!), tol: 1e-3)
    check("unit.codec_resunit_out",
          codec.residualUnit(toNLC(goldens["unit.x_res_in"]!), prefix: "encoder.block.1.block.0", dilation: 1),
          toNLC(goldens["unit.codec_resunit_out"]!), tol: 1e-4)
    check("unit.codec_pre_module_out",
          codec.windowTransformer(toNLC(goldens["unit.x_wt_in"]!), prefix: "quantizer.pre_module",
                                  spec: codec.preSpec),
          toNLC(goldens["unit.codec_pre_module_out"]!), tol: 1e-3)
    let codesIn = goldens["unit.codes_in"]!.asType(.int32)
    check("unit.codec_qdecode_out", codec.quantizerDecode(codesIn),
          toNLC(goldens["unit.codec_qdecode_out"]!), tol: 1e-3)
    check("unit.codec_full_decode_out", codec.decode(codesIn).expandedDimensions(axis: 1).transposed(0, 2, 1),
          toNLC(goldens["unit.codec_full_decode_out"]!), tol: 1e-3)
    finish("S1")
}

/// CPU-exact parity: codec encode, the composite embedding, and prefill.
///
/// For `falcon_h1` this stops before the generation rollout, which cannot run here:
/// mlx-swift-lm's cached Mamba step is a `metal_kernel` and hard-errors on the CPU stream
/// ("[metal_kernel] Only supports the GPU"). Switching the default device mid-run does not
/// help — the default stream is already resolved — so the rollout lives in its own process
/// as `--s2gen`. Keeping the split at the process boundary is what preserves the exactness
/// of everything above it: running the whole gate on the GPU instead drops codec encode
/// from 100% to 95.6% code-exact and prefill from 1.8e-05 to 2.8e-02, because GPU fp32
/// reorders reductions and the VQ nearest-neighbour argmax flips on ties.
func gateS2() throws {
    Device.setDefault(device: Device(.cpu))
    let goldens = loadGoldens()
    let model = try Audio8TTS.load(directory: modelDir, lmDtype: .float32)
    let lm = model.lm

    // codec encode of the real reference clip: 100% code-exact
    let refAudio = goldens["ref_audio_44100"]!
    let refLen = goldens["proc.reference_audio_lengths"]!.asType(.int32).item(Int32.self)
    let (refCodes, refCodeLen) = model.codec.encode(refAudio.expandedDimensions(axis: 0), sampleCount: Int(refLen))
    eval(refCodes)
    let goldenCodes = goldens["encode.ref_codes"]!.asType(.int32)
    let agree = (refCodes .== goldenCodes).asType(.float32).mean().item(Float.self)
    print("  \(agree == 1.0 ? "ok  " : "FAIL") encode.ref_codes agreement \(agree * 100)%")
    if agree < 1.0 { failures.append("encode.ref_codes agreement \(agree)") }

    // prefill parity on the prompt fixture
    let prompt = goldens["prompt.ids"]!.asType(.int32)
    let promptMask = goldens["prompt.mask"]!.asType(.int32)
    // Same tensor, different fixture name per capture: the 0.1b harness calls it
    // `embed.raw_unscaled` because that path has a second, SCALED embedding worth pinning
    // separately (the multiplier belongs on the composite, not the lookup table).
    let embedKey = lm.config.usesFalconSlow ? "embed.raw_unscaled" : "prefill.embed"
    check(embedKey, lm.embed(prompt), goldens[embedKey]!, tol: 1e-4)
    let (logits, hidden) = lm(prompt, attentionMask: promptMask)
    check("prefill.logits_last", logits[0..., -1], goldens["prefill.logits_last"]!, tol: 2e-2)
    check("prefill.hidden_last", hidden[0..., -1], goldens["prefill.hidden_last"]!, tol: 2e-2)

    if lm.config.usesFalconSlow {
        print("  --   greedy rollout skipped on CPU (GPU-only cached Mamba step) — run --s2gen")
        finish("S2")
        return
    }

    // greedy generation: token-exact vs the oracle
    let prefix: [Int32] = goldens["proc.prefix_input_ids"]![0].asArray(Int32.self)
    let suffix: [Int32] = goldens["proc.suffix_input_ids"]![0].asArray(Int32.self)
    let generated = try lm.generateCodes(
        prefix: prefix, suffix: suffix,
        referenceCodes: refCodes[0], referenceLength: refCodeLen,
        params: SamplingParams(maxNewTokens: 200, doSample: false))
    eval(generated)
    let goldenGreedy = goldens["gen.codes_greedy"]!.asType(.int32)
    print("  greedy codes: swift \(generated.shape) vs golden \(goldenGreedy.shape)")
    if generated.shape == goldenGreedy.shape {
        let match = (generated .== goldenGreedy).asType(.float32).mean().item(Float.self)
        print("  \(match == 1.0 ? "ok  " : "FAIL") greedy code agreement \(match * 100)%")
        if match < 1.0 { failures.append("greedy agreement \(match)") }
    } else {
        failures.append("greedy shape \(generated.shape) vs \(goldenGreedy.shape)")
    }

    // decode the golden codes -> waveform parity
    let wave = model.codec.decode(goldenGreedy)
    eval(wave)
    let goldenWave = goldens["gen.waveform"]![0]
    let got = wave[0][..<goldenWave.shape[0]]
    let diff = abs(got - goldenWave).max().item(Float.self)
    print("  \(diff < 5e-3 ? "ok  " : "FAIL") waveform max_abs \(diff)")
    if diff >= 5e-3 { failures.append("waveform diff \(diff)") }
    finish("S2")
}

/// The generation rollout for `falcon_h1`, on the GPU because it has to be (see `gateS2`).
///
/// Deliberately NOT a token-parity proof, and the code-agreement line below is reported as
/// information rather than checked. A GPU rollout CANNOT be token-exact against a CPU
/// oracle here, and the reason is upstream of the language model: the codec's VQ
/// nearest-neighbour encode is not code-exact across devices, so the reference codes differ,
/// the prompt differs, and every frame after that is conditioned differently.
///
/// Measured, same checkpoint, same reference clip, deterministic decoding:
///
///     runtime            codec encode    greedy agreement vs the CPU oracle
///     Python-MLX  CPU        100%              100%  (token-exact)
///     Python-MLX  GPU       94.95%             7.76%
///     Swift       GPU       95.59%            12.06%
///
/// Swift agrees slightly BETTER than Python-MLX does on the same device, so a low number
/// here is a device effect, not a defect in this port. Token-exactness for this checkpoint
/// is established CPU-to-CPU (gateS2 above, and parity_mlx.py at 100% over 107 frames).
/// What this gate adds is the one thing those cannot reach: that the Swift cached-Mamba
/// path runs at all and produces audio of the right length and level.
func gateS2Gen() throws {
    Device.setDefault(device: Device(.gpu))
    let goldens = loadGoldens()
    let model = try Audio8TTS.load(directory: modelDir, lmDtype: .float32)
    let lm = model.lm
    guard lm.config.usesFalconSlow else {
        print("  --   --s2gen is for the falcon_h1 checkpoint; --s2 already covers arktts")
        finish("S2gen")
        return
    }

    let refAudio = goldens["ref_audio_44100"]!
    let refLen = goldens["proc.reference_audio_lengths"]!.asType(.int32).item(Int32.self)
    let (refCodes, refCodeLen) = model.codec.encode(
        refAudio.expandedDimensions(axis: 0), sampleCount: Int(refLen))
    eval(refCodes)

    let prefix: [Int32] = goldens["proc.prefix_input_ids"]![0].asArray(Int32.self)
    let suffix: [Int32] = goldens["proc.suffix_input_ids"]![0].asArray(Int32.self)
    let generated = try lm.generateCodes(
        prefix: prefix, suffix: suffix,
        referenceCodes: refCodes[0], referenceLength: refCodeLen,
        params: SamplingParams(maxNewTokens: 200, doSample: false))
    eval(generated)

    let goldenGreedy = goldens["gen.codes_greedy"]!.asType(.int32)
    let frames = generated.shape[2]
    let goldenFrames = goldenGreedy.shape[2]
    print("  --   greedy frames: swift(GPU) \(frames) vs oracle(CPU) \(goldenFrames)")
    // Length within a frame or two, and the shared prefix mostly agreeing, is what a
    // correct-but-not-bit-identical rollout looks like. A broken one diverges immediately.
    let lengthOK = abs(frames - goldenFrames) <= 2
    print("  \(lengthOK ? "ok  " : "FAIL") rollout length within 2 frames of the oracle")
    if !lengthOK { failures.append("greedy length \(frames) vs \(goldenFrames)") }

    let shared = min(frames, goldenFrames)
    let agree = (generated[0..., 0..., ..<shared] .== goldenGreedy[0..., 0..., ..<shared])
        .asType(.float32).mean().item(Float.self)
    // Informational. See this function's doc comment: cross-device VQ encode makes a high
    // number impossible, so this is NOT a pass/fail check. Python-MLX on the same device
    // scores 7.76% against the same oracle.
    print("  --   code agreement over the shared \(shared) frames: \(agree * 100)% "
        + "(informational — cross-device; python-mlx GPU scores 7.76% here)")

    // The load-bearing check: the audio must be real speech, not silence or noise.
    let wave = model.codec.decode(generated)
    eval(wave)
    let audio = wave[0]
    let rms = sqrt((audio * audio).mean()).item(Float.self)
    let peak = abs(audio).max().item(Float.self)
    let dB = { (x: Float) in 20 * log10(max(x, 1e-12)) }
    let seconds = Float(audio.shape[0]) / Float(lm.config.codecSampleRate)
    print("  --   audio \(seconds)s  rms \(dB(rms)) dBFS  peak \(dB(peak)) dBFS")
    let audible = dB(rms) > -40 && dB(rms) < -3 && peak <= 1.0
    print("  \(audible ? "ok  " : "FAIL") decoded audio is in a sane level range")
    if !audible { failures.append("audio level rms \(dB(rms)) dBFS peak \(dB(peak)) dBFS") }

    let wavURL = goldensURL.deletingLastPathComponent()
        .appendingPathComponent("swift_smoke_0.1b_gpu.wav")
    try writeWav(audio.asArray(Float.self), to: wavURL, sampleRate: lm.config.codecSampleRate)
    print("  --   wav -> \(wavURL.path)")
    finish("S2gen")
}

func gateS2b() async throws {
    let goldens = loadGoldens()
    let model = try Audio8TTS.load(directory: modelDir)  // bf16 LM, GPU stream

    // prompt ids must match the processor fixtures token-for-token
    let refText = "You know, I was thinking about what you said earlier, and honestly, I think you might be right about the whole thing."
    let targetText = "The quick brown fox jumps over the lazy dog, while the river keeps flowing north."
    let (prefix, suffix) = try await model.promptSegments(
        text: targetText, referenceText: refText, hasReference: true)
    let goldPrefix: [Int32] = goldens["proc.prefix_input_ids"]![0].asArray(Int32.self)
    let goldSuffix: [Int32] = goldens["proc.suffix_input_ids"]![0].asArray(Int32.self)
    let prefixOK = prefix == goldPrefix
    let suffixOK = suffix == goldSuffix
    print("  \(prefixOK ? "ok  " : "FAIL") prompt prefix ids (\(prefix.count) tokens)")
    print("  \(suffixOK ? "ok  " : "FAIL") prompt suffix ids (\(suffix.count) tokens)")
    if !prefixOK { failures.append("prefix ids mismatch") }
    if !suffixOK { failures.append("suffix ids mismatch") }

    // one real sampled generation on the GPU
    let refAudio = goldens["ref_audio_44100"]!
    let result = try await model.generate(
        text: "Swift port smoke test. The dual A R model and codec are running on the GPU.",
        referenceAudio: refAudio,
        referenceText: refText,
        params: SamplingParams(maxNewTokens: 300, seed: 1234))
    let samples: [Float] = result.audio.asArray(Float.self)
    let rms = (samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count)).squareRoot()
    let dbfs = 20 * log10(rms + 1e-12)
    print(String(format: "  frames %d  dur %.2fs  RTF %.2f  rms %.1f dBFS",
                 result.frames, result.duration, result.realTimeFactor, dbfs))
    if dbfs < -60 { failures.append("silent output (\(dbfs) dBFS)") }
    let wavURL = workspaceRoot.appendingPathComponent("oracle-capture/goldens/swift_smoke_bf16.wav")
    try writeWav(samples, to: wavURL)
    print("  wav -> \(wavURL.path)")
    finish("S2b")
}

// MARK: - S7 / --validate: the ENGINE path, headless, with a measured footprint

/// Real-process resident memory (mach phys_footprint) in MB — the app-truth footprint.
/// MLX's own GPU peak under-reads vs phys, so both are reported.
func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576.0 : -1
}

func mb(_ bytes: Int) -> Double { Double(bytes) / 1_048_576.0 }

/// Drives the real `Audio8Package` through register-equivalent lifecycle (construct → load →
/// run → unload) and reports the numbers the manifest must declare. This is the measurement
/// of record for `bf16ResidentBytes` / `peakActivationBytes`.
func gateValidate() async throws {
    let goldens = loadGoldens()
    let refText = "You know, I was thinking about what you said earlier, and honestly, I think you might be right about the whole thing."

    // A reference clip as the canonical Audio artifact the engine would hand us.
    let refSamples: [Float] = goldens["ref_audio_44100"]!.asArray(Float.self)
    let refWavURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("audio8-validate-ref.wav")
    try writeWav(refSamples, to: refWavURL)
    let refAudio = Audio(format: .wav, data: try Data(contentsOf: refWavURL),
                         sampleRate: 44100, channels: 1)

    let baseline = physFootprintMB()
    MLX.Memory.clearCache()
    let package = Audio8Package(
        configuration: Audio8Configuration(modelDirectory: modelDir))

    // --- load ---
    let loadStart = Date()
    try await package.load()
    let loadSeconds = Date().timeIntervalSince(loadStart)
    // Resident floor is measured POST-LOAD, before any run allocates a transient.
    let residentActive = MLX.Memory.activeMemory
    let residentPhys = physFootprintMB()
    GPU.resetPeakMemory()

    // --- run (through the engine-facing surface, not the core) ---
    let request = TTSRequest(
        text: "Validation run through the engine package surface. This sentence is long enough "
            + "to exercise a realistic single-utterance transient in the codec decoder.",
        voice: VoiceSelector(.referenceAudio(refAudio)),
        referenceTranscript: refText,
        metaData: ["seed": .int(4242), "maxFrames": .int(400)])
    let runStart = Date()
    let response = try await package.run(request)
    let runSeconds = Date().timeIntervalSince(runStart)
    let peakActive = MLX.Memory.peakMemory
    let peakPhys = physFootprintMB()

    guard let tts = response as? TTSResponse else {
        failures.append("run() did not return a TTSResponse")
        return finish("VALIDATE")
    }

    // Quantify the audio — a silent stem reads −∞ dBFS; never trust ears alone.
    let pcm = tts.audio.data.dropFirst(44)
    let channels = tts.audio.channels ?? 1
    var decoded = [Float]()
    decoded.reserveCapacity(pcm.count / 2)
    var iterator = pcm.makeIterator()
    while let low = iterator.next(), let high = iterator.next() {
        let sample = Int16(bitPattern: UInt16(low) | (UInt16(high) << 8))
        decoded.append(Float(sample) / 32767)
    }
    let rms = (decoded.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(decoded.count, 1))).squareRoot()
    let dbfs = 20 * log10(rms + 1e-12)
    let sampleRate = tts.audio.sampleRate ?? ArkttsCodec.sampleRate
    let duration = Double(decoded.count) / Double(sampleRate)

    print("  ok   response: .\(tts.audio.format) \(sampleRate) Hz "
          + "\(channels)ch, \(decoded.count) samples")
    print(String(format: "  %@ audio level %.1f dBFS over %.2fs",
                 dbfs > -60 ? "ok  " : "FAIL", dbfs, duration))
    if dbfs <= -60 { failures.append("silent output (\(dbfs) dBFS)") }
    print(String(format: "  load %.2fs · run %.2fs · RTF %.2f", loadSeconds, runSeconds,
                 duration > 0 ? runSeconds / duration : 0))

    // --- the footprint table (what the manifest must declare) ---
    print("")
    print(String(format: "  baseline phys           %8.0f MB", baseline))
    print(String(format: "  RESIDENT  MLX-active    %8.0f MB   phys %.0f MB",
                 mb(residentActive), residentPhys))
    print(String(format: "  PEAK      MLX-high-water%8.0f MB   phys %.0f MB",
                 mb(peakActive), peakPhys))
    let transient = max(0, mb(peakActive) - mb(residentActive))
    print(String(format: "  transient (peak − resident) %6.0f MB", transient))
    print("")
    // NOTE the split-footprint semantics: `peakActivationBytes` is the TRANSIENT (peak minus
    // resident), not the absolute high-water — the engine adds it to the resident floor when
    // admitting. Declaring the absolute peak here would double-count the weights.
    print("  DECLARE → residentBytes: \(UInt64(Double(residentActive) * 1.07))")
    print("            peakActivationBytes: \(UInt64(transient * 1_048_576 * 1.13))"
          + "   (= transient, NOT absolute peak)")
    print("  (resident ×1.07, transient ×1.13 margin — the fleet convention)")

    // --- unload must actually release ---
    // Attribute what remains: MLX's own accounting (active = live arrays, cache = the buffer
    // pool MLX retains for reuse) vs process phys. A large phys with near-zero MLX-active is the
    // allocator/pool holding pages, NOT a package leak — the distinction the fleet's
    // "memory never releases" reports keep tripping over.
    await package.unload()
    let afterUnload = physFootprintMB()
    print(String(format: "\n  unload → phys %.0f MB (released %.0f MB)",
                 afterUnload, peakPhys - afterUnload))
    print(String(format: "  post-unload MLX: active %.0f MB · cache %.0f MB · peak %.0f MB",
                 mb(MLX.Memory.activeMemory), mb(MLX.Memory.cacheMemory),
                 mb(MLX.Memory.peakMemory)))
    if mb(MLX.Memory.activeMemory) > 256 {
        failures.append(String(format: "MLX still holds %.0f MB active after unload",
                               mb(MLX.Memory.activeMemory)))
    }

    // --- transient scaling: the peak is utterance-length-driven, so show the slope ---
    GPU.resetPeakMemory()
    let shortPackage = Audio8Package(
        configuration: Audio8Configuration(modelDirectory: modelDir))
    try await shortPackage.load()
    let shortResident = MLX.Memory.activeMemory
    GPU.resetPeakMemory()
    let shortRequest = TTSRequest(
        text: "A short line.",
        voice: VoiceSelector(.referenceAudio(refAudio)),
        referenceTranscript: refText,
        metaData: ["seed": .int(4242), "maxFrames": .int(60)])
    _ = try await shortPackage.run(shortRequest)
    print(String(format: "\n  short utterance: resident %.0f MB · peak %.0f MB (transient %.0f MB)",
                 mb(shortResident), mb(MLX.Memory.peakMemory),
                 mb(MLX.Memory.peakMemory) - mb(shortResident)))
    await shortPackage.unload()
    finish("VALIDATE")
}


// MARK: - S5: windowed decode must be BIT-IDENTICAL to the whole-utterance decode

/// The streaming decode is only worth having if it is the SAME computation, chunked — not an
/// approximation. This gate proves that: decode a real utterance whole, decode it again in
/// chunks, and require max_abs == 0 exactly. It also runs a deliberately-too-small context to
/// show the gate can fail — a bit-identity check that cannot fail is not evidence.
func gateS5() throws {
    // Stream selection is the variable under test here, so it is set explicitly:
    // AUDIO8_S5_CPU=1 pins the CPU stream to separate ALGORITHMIC exactness from
    // Metal's input-size-dependent kernel selection.
    let onCPU = ProcessInfo.processInfo.environment["AUDIO8_S5_CPU"] == "1"
    if onCPU { Device.setDefault(device: Device(.cpu)) }
    print("  stream: \(onCPU ? "CPU" : "GPU")")
    let goldens = loadGoldens()
    let model = try Audio8TTS.load(directory: modelDir, lmDtype: .float32)
    let codec = model.codec
    let codes = goldens["gen.codes_greedy"]!.asType(.int32)
    let frames = codes.shape[2]
    let L = codec.leftReceptiveFieldFrames
    print("  left receptive field: \(L) frames (\(String(format: "%.2f", Double(L) * 0.0464)) s)")

    // Reference: one whole-utterance decode.
    let whole = codec.decode(codes)
    eval(whole)
    let wholeSamples = whole.shape[1]

    func streamedWaveform(chunkFrames: Int, context: Int?) throws -> MLXArray {
        var pieces: [MLXArray] = []
        try codec.decodeStreaming(codes, chunkFrames: chunkFrames, contextFrames: context) { s, _ in
            pieces.append(s)
        }
        return concatenated(pieces, axis: 1)
    }

    // Exactness vs CHUNK SIZE. The context sweep below shows context is not the variable;
    // window SIZE is, because MLX selects different conv kernels for small inputs and their
    // fp32 reduction order differs from the whole-utterance path.
    let floor = codec.minimumExactChunkFrames
    for chunk in [8, 16, 32, 48, 64, 96, 128] {
        let streamed = try streamedWaveform(chunkFrames: chunk, context: nil)
        eval(streamed)
        guard streamed.shape[1] == wholeSamples else {
            failures.append("S5 chunk \(chunk): length \(streamed.shape[1]) vs \(wholeSamples)")
            print("  FAIL chunk \(chunk): length mismatch")
            continue
        }
        let diff = abs(whole - streamed).max().item(Float.self)
        if chunk >= floor {
            // At or above the measured floor the contract is EXACT equality.
            if diff != 0 { failures.append("S5 chunk \(chunk): max_abs \(diff) (must be exactly 0)") }
            print("  \(diff == 0 ? "ok  " : "FAIL") chunk \(chunk) frames: max_abs \(diff)  [exact]")
        } else {
            // Below it, drift is expected (small-input conv kernels) but must stay bounded and
            // inaudible. Recorded rather than ignored — an unbounded drift here would be a bug.
            let bounded = diff <= 2e-3
            if !bounded { failures.append("S5 chunk \(chunk): sub-floor drift \(diff) exceeds 2e-3") }
            print("  \(bounded ? "note" : "FAIL") chunk \(chunk) frames: max_abs \(diff)  "
                  + "[below floor \(floor); drift expected, bounded]")
        }
    }

    // Empirical context sweep: find the SMALLEST context that is exact at a small chunk size.
    // The computed value is a claim; this is the measurement that settles it.
    // Run this ABOVE the chunk floor, otherwise kernel-selection drift masks the very thing
    // being measured (the first version of this sweep ran at chunk 8 and could never be exact,
    // for reasons that had nothing to do with context).
    print("  context sweep (chunk 64 — above the exactness floor):")
    var minimumExact: Int?
    for ctx in [0, 4, 8, 10, 11, 12, 16, 32] {
        let s = try streamedWaveform(chunkFrames: 64, context: ctx)
        eval(s)
        let d = s.shape[1] == wholeSamples ? abs(whole - s).max().item(Float.self) : Float.nan
        print("    L=\(ctx): max_abs \(d)")
        if d == 0, minimumExact == nil { minimumExact = ctx }
    }
    if let minimumExact {
        // The contract on a receptive-field bound is SUFFICIENCY, not tightness: the derived
        // value must be >= the measured minimum. Being conservative costs a little compute per
        // chunk; being short silently corrupts the seam. (Derived 11 vs measured 10 — the ceil()
        // chain rounds up once more than strictly needed, which is the direction to err in.)
        let sufficient = L >= minimumExact
        print("  \(sufficient ? "ok  " : "FAIL") smallest exact context: \(minimumExact) frames; "
              + "derived \(L) (\(sufficient ? "sufficient, +\(L - minimumExact) slack" : "TOO SHORT"))")
        if !sufficient {
            failures.append("S5: derived context \(L) is SHORT of the measured minimum "
                            + "\(minimumExact) — windowed decode is not exact")
        }
    } else {
        failures.append("S5: no context in the swept range produced an exact decode")
    }

    // Negative control: too little context MUST drift, otherwise the check above proves nothing.
    let starved = try streamedWaveform(chunkFrames: 32, context: 0)
    eval(starved)
    let starvedDiff = abs(whole - starved).max().item(Float.self)
    let drifts = starvedDiff > 0
    if !drifts {
        failures.append("S5 negative control: zero context produced an identical result — "
                        + "the bit-identity gate is not actually testing anything")
    }
    print("  \(drifts ? "ok  " : "FAIL") negative control (0 context) drifts: max_abs \(starvedDiff)")

    finish("S5")
}


// MARK: - STREAM live: does windowed decode actually bound memory and emit early?

/// S5 proves the streamed waveform is bit-identical. This proves it is WORTH having: measures
/// time-to-first-audio and peak activation for the streaming path against the batch path on the
/// same request, through the real engine package.
func gateStream() async throws {
    let goldens = loadGoldens()
    let refText = "You know, I was thinking about what you said earlier, and honestly, I think you might be right about the whole thing."
    let refSamples: [Float] = goldens["ref_audio_44100"]!.asArray(Float.self)
    let refURL = FileManager.default.temporaryDirectory.appendingPathComponent("audio8-stream-ref.wav")
    try writeWav(refSamples, to: refURL)
    let refAudio = Audio(format: .wav, data: try Data(contentsOf: refURL),
                         sampleRate: 44100, channels: 1)

    let package = try makeAudio8Package(modelDir)
    try await package.load()

    let text = "This passage is long enough that streaming has something to prove: audio should "
        + "start playing well before the last frame is generated, and the decoder's activation "
        + "peak should stay bounded by the chunk rather than growing with the utterance."
    func request() -> TTSRequest {
        TTSRequest(text: text,
                   voice: VoiceSelector(.referenceAudio(refAudio)),
                   referenceTranscript: refText,
                   metaData: ["seed": .int(11), "greedy": .bool(true), "maxFrames": .int(400)])
    }

    // Batch
    GPU.resetPeakMemory()
    let floor = MLX.Memory.activeMemory
    let batchStart = Date()
    let batchResponse = try await package.run(request())
    let batchSeconds = Date().timeIntervalSince(batchStart)
    let batchPeak = MLX.Memory.peakMemory - floor
    let batchBytes = (batchResponse as? TTSResponse)?.audio.data.count ?? 0

    // Streaming
    GPU.resetPeakMemory()
    var firstAudioAt: TimeInterval?
    var chunks = 0
    let streamStart = Date()
    let streamResponse = try await package.runStream(request()) { chunk in
        if firstAudioAt == nil { firstAudioAt = Date().timeIntervalSince(streamStart) }
        chunks += 1
        _ = chunk.isFinal
    }
    let streamSeconds = Date().timeIntervalSince(streamStart)
    let streamPeak = MLX.Memory.peakMemory - floor
    let streamBytes = (streamResponse as? TTSResponse)?.audio.data.count ?? 0

    func mb(_ v: Int) -> Double { Double(v) / 1_048_576 }
    print(String(format: "  batch:  %.2f s   peak +%.0f MB", batchSeconds, mb(batchPeak)))
    print(String(format: "  stream: %.2f s   peak +%.0f MB   %d chunks   TTFA %.2f s",
                 streamSeconds, mb(streamPeak), chunks, firstAudioAt ?? -1))

    // STR-5: the aggregated streaming response must match what run() produced — by CONTENT,
    // not by size. Comparing byte counts is a gate that cannot fail on the interesting bug:
    // an approximate decode produces exactly as many samples as an exact one. (It did, and this
    // check is why that was caught.)
    let batchData = (batchResponse as? TTSResponse)?.audio.data ?? Data()
    let streamData = (streamResponse as? TTSResponse)?.audio.data ?? Data()
    if batchData == streamData {
        print("  ok   aggregated response byte-identical to batch (\(batchBytes) bytes)")
    } else if batchBytes != streamBytes {
        failures.append("stream aggregate \(streamBytes) bytes != batch \(batchBytes)")
        print("  FAIL length differs: \(streamBytes) vs \(batchBytes)")
    } else {
        let differing = zip(batchData, streamData).filter { $0 != $1 }.count
        failures.append("stream aggregate same length but \(differing) bytes differ from batch")
        print("  FAIL same length, \(differing) bytes differ — the streamed decode is not exact")
    }
    // The point of the feature.
    // Both paths are windowed now, so the meaningful assertion is that each is BOUNDED —
    // parity between them is the expected outcome, not a failure. (An earlier version demanded
    // streaming < batch, which stopped being the right question once batch was fixed too.)
    // Read the bound from the package under test, not from a constant. Hardcoding the 0.6b's
    // 4.2 GB meant this gate asserted against a number Audio8MiniPackage does not declare —
    // it happened to pass, which is exactly how a gate stops checking anything.
    let bound = Int(type(of: package).manifest.requirements.footprints
        .first { $0.quant == .bf16 }?.peakActivationBytes ?? 4_200_000_000)
    let bothBounded = batchPeak < bound && streamPeak < bound
    print(String(format: "  %@ both paths within the declared %.1f GB envelope "
                 + "(batch %.0f MB, stream %.0f MB)",
                 bothBounded ? "ok  " : "FAIL", Double(bound) / 1e9,
                 mb(batchPeak), mb(streamPeak)))
    if !bothBounded { failures.append("a decode path exceeded the declared activation envelope") }
    if let ttfa = firstAudioAt, ttfa < batchSeconds {
        print(String(format: "  ok   first audio %.2f s ahead of the batch result", batchSeconds - ttfa))
    } else {
        failures.append("no time-to-first-audio advantage")
    }
    await package.unload()
    finish("STREAM")
}


// MARK: - STREAMENV: does streaming actually BOUND activation, or merely reduce it?

/// The batch transient is linear in frames (1824 + 14.2·frames MB). Streaming should be FLAT —
/// bounded by chunk+context, not by utterance length. That is the difference between "uses less
/// memory" and "has a declarable envelope", so it is measured across the length range rather
/// than asserted from one run.
func gateStreamEnvelope() async throws {
    let goldens = loadGoldens()
    let refText = "You know, I was thinking about what you said earlier, and honestly, I think you might be right about the whole thing."
    let refSamples: [Float] = goldens["ref_audio_44100"]!.asArray(Float.self)
    let refURL = FileManager.default.temporaryDirectory.appendingPathComponent("audio8-env-ref.wav")
    try writeWav(refSamples, to: refURL)
    let refAudio = Audio(format: .wav, data: try Data(contentsOf: refURL),
                         sampleRate: 44100, channels: 1)

    let package = try makeAudio8Package(modelDir)
    try await package.load()
    MLX.GPU.clearCache()
    let floor = MLX.Memory.activeMemory
    print(String(format: "  resident floor %.0f MB", Double(floor) / 1_048_576))
    print("  frames   batch peak      stream peak")

    // maxFrames drives utterance length; greedy keeps it deterministic across the two paths.
    for cap in [64, 128, 256, 512] {
        func request() -> TTSRequest {
            TTSRequest(text: "This passage exists to be generated at a controlled length so the "
                       + "activation envelope can be measured against the number of frames the "
                       + "model actually produces, rather than against a single sample.",
                       voice: VoiceSelector(.referenceAudio(refAudio)),
                       referenceTranscript: refText,
                       metaData: ["seed": .int(5), "greedy": .bool(true), "maxFrames": .int(cap)])
        }
        MLX.GPU.clearCache(); GPU.resetPeakMemory()
        _ = try await package.run(request())
        let batchPeak = MLX.Memory.peakMemory - floor

        MLX.GPU.clearCache(); GPU.resetPeakMemory()
        var frames = 0
        _ = try await package.runStream(request()) { chunk in
            frames += chunk.samples.count / ArkttsCodec.frameLength
        }
        let streamPeak = MLX.Memory.peakMemory - floor
        print(String(format: "  %5d    %7.0f MB      %7.0f MB",
                     frames, Double(batchPeak) / 1_048_576, Double(streamPeak) / 1_048_576))
    }
    await package.unload()
    finish("STREAMENV")
}

// MARK: - CAN live: the MID-RUN checkpoint (the offline gate only proves the entry one)

/// The offline `CancellationConformance.checkRun` cancels BEFORE `run()` starts, so it exercises
/// the entry checkpoint only. This probe lets a real generation get underway, cancels it in
/// flight, and asserts (a) it stops promptly rather than running to completion, and (b) the error
/// surfaced is a `CancellationError` — unwrapped — so the engine can classify it.
func gateCancel() async throws {
    let goldens = loadGoldens()
    let refText = "You know, I was thinking about what you said earlier, and honestly, I think you might be right about the whole thing."
    let refSamples: [Float] = goldens["ref_audio_44100"]!.asArray(Float.self)
    let refWavURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("audio8-cancel-ref.wav")
    try writeWav(refSamples, to: refWavURL)
    let refAudio = Audio(format: .wav, data: try Data(contentsOf: refWavURL),
                         sampleRate: 44100, channels: 1)

    let package = try makeAudio8Package(modelDir)
    try await package.load()

    // A long request, so an uncancelled run would take many seconds.
    let request = TTSRequest(
        text: "This is a deliberately long utterance used to prove that cancellation interrupts "
            + "the autoregressive rollout mid-flight rather than only at the entry checkpoint, "
            + "which is all the offline conformance gate can demonstrate on its own.",
        voice: VoiceSelector(.referenceAudio(refAudio)),
        referenceTranscript: refText,
        metaData: ["seed": .int(7), "maxFrames": .int(600)])

    let start = Date()
    let task = Task { @InferenceActor in try await package.run(request) }
    // Let the rollout actually start, then cancel.
    try await Task.sleep(for: .milliseconds(1500))
    task.cancel()
    let outcome = await task.result
    let elapsed = Date().timeIntervalSince(start)

    switch outcome {
    case .failure(is CancellationError):
        print(String(format: "  ok   mid-run cancel surfaced CancellationError after %.2fs", elapsed))
        // An uncancelled 600-frame run is ~25 s of audio and >20 s of compute; stopping in a
        // few seconds is the evidence the per-frame checkpoint fired.
        if elapsed > 10 {
            failures.append(String(format: "cancel took %.1fs — checkpoint cadence too coarse", elapsed))
        } else {
            print(String(format: "  ok   stopped promptly (%.2fs ≪ full-run time)", elapsed))
        }
    case .failure(let error):
        failures.append("cancel surfaced \(type(of: error)) — must be an unwrapped CancellationError")
        print("  FAIL cancel surfaced \(type(of: error)): \(error)")
    case .success:
        failures.append("run completed despite cancellation — no mid-run checkpoint")
        print("  FAIL run ran to completion despite cancellation")
    }
    await package.unload()
    finish("CANCEL")
}

// MARK: entry

let mode = CommandLine.arguments.dropFirst().first ?? "--s1"
stderrPrint("audio8-gates \(mode)  model=\(modelDir.path)")
switch mode {
case "--s0": try gateS0()
case "--s5": try gateS5()
case "--s1": try gateS1()
case "--s2": try gateS2()
case "--s2gen": try gateS2Gen()
case "--s2b", "--validate", "--cancel", "--stream", "--streamenv":
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            switch mode {
            case "--s2b": try await gateS2b()
            case "--cancel": try await gateCancel()
            case "--stream": try await gateStream()
            case "--streamenv": try await gateStreamEnvelope()
            default: try await gateValidate()
            }
        } catch {
            stderrPrint("\(mode) error: \(error)")
            exit(1)
        }
        semaphore.signal()
    }
    semaphore.wait()
default:
    stderrPrint("unknown mode \(mode); use --s0 | --s1 | --s2 | --s2b | --s5 | --validate | --cancel | --stream")
    exit(2)
}
