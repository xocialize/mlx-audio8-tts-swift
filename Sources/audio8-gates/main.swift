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
    let generated = lm.generateCodes(
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

// MARK: entry

let mode = CommandLine.arguments.dropFirst().first ?? "--s1"
stderrPrint("audio8-gates \(mode)  model=\(modelDir.path)")
switch mode {
case "--s0": try gateS0()
case "--s1": try gateS1()
case "--s2": try gateS2()
case "--s2b":
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do { try await gateS2b() } catch {
            stderrPrint("S2b error: \(error)")
            exit(1)
        }
        semaphore.signal()
    }
    semaphore.wait()
default:
    stderrPrint("unknown mode \(mode); use --s0 | --s1 | --s2 | --s2b")
    exit(2)
}
