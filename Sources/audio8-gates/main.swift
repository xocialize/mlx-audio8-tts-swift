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
import MLX
import Audio8TTSCore
import MLXAudio8TTS
import MLXToolKit

// MARK: paths

let fileDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let repoRoot = fileDir.deletingLastPathComponent().deletingLastPathComponent()
let workspaceRoot = repoRoot.deletingLastPathComponent()
let goldensURL = workspaceRoot.appendingPathComponent("oracle-capture/goldens/arktts_goldens.safetensors")
let modelDir = ProcessInfo.processInfo.environment["AUDIO8_MODEL_DIR"].map { URL(fileURLWithPath: $0) }
    ?? workspaceRoot.appendingPathComponent("release/Audio8-TTS-Preview-0.6b-bf16")

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

    check("unit.rope_table_slow", lm.freqsCis.values[..<32], goldens["unit.rope_table_slow"]!, tol: 1e-6)
    check("unit.rope_table_fast", lm.fastFreqsCis.values, goldens["unit.rope_table_fast"]!, tol: 1e-6)

    let xSlow = goldens["unit.x_slow_in"]!
    let length = 17
    let rope = lm.freqsCis.values[MLXArray(0..<Int32(length))].expandedDimensions(axis: 0)
    let row = MLXArray(0..<Int32(length))[0..., .newAxis]
    let col = MLXArray(0..<Int32(length))[.newAxis, 0...]
    let mask = (col .<= row).expandedDimensions(axes: [0, 1])
    check("unit.rmsnorm_out", lm.norm(xSlow), goldens["unit.rmsnorm_out"]!, tol: 1e-5)
    check("unit.slow_ffn0_out", lm.layers[0].feedForward(xSlow), goldens["unit.slow_ffn0_out"]!, tol: 1e-4)
    check(
        "unit.slow_attn0_out",
        lm.layers[0].attention(lm.layers[0].attentionNorm(xSlow), rope: rope, mask: mask),
        goldens["unit.slow_attn0_out"]!, tol: 1e-4)
    check("unit.slow_block0_out", lm.layers[0](xSlow, rope: rope, mask: mask),
          goldens["unit.slow_block0_out"]!, tol: 1e-4)
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
    check("prefill.embed", lm.embed(prompt), goldens["prefill.embed"]!, tol: 1e-4)
    let (logits, hidden) = lm(prompt, attentionMask: promptMask)
    check("prefill.logits_last", logits[0..., -1], goldens["prefill.logits_last"]!, tol: 2e-2)
    check("prefill.hidden_last", hidden[0..., -1], goldens["prefill.hidden_last"]!, tol: 2e-2)

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

    let package = Audio8Package(configuration: Audio8Configuration(modelDirectory: modelDir))
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
case "--s1": try gateS1()
case "--s2": try gateS2()
case "--s2b", "--validate", "--cancel":
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            switch mode {
            case "--s2b": try await gateS2b()
            case "--cancel": try await gateCancel()
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
    stderrPrint("unknown mode \(mode); use --s0 | --s1 | --s2 | --s2b | --validate | --cancel")
    exit(2)
}
