// Audio8TTS.swift — top-level model: loader (S0 key contract enforced), prompt
// builder (Qwen chat format + <|voice|> trigger), and the text→waveform surface.

import Foundation
import MLX
import MLXNN
import Tokenizers

public struct Audio8GenerationResult {
    public let audio: MLXArray          // (samples,) fp32 44.1 kHz
    public let frames: Int
    public let sampleRate: Int
    public let elapsed: TimeInterval
    public var duration: Double { Double(audio.shape[0]) / Double(sampleRate) }
    public var realTimeFactor: Double { duration > 0 ? elapsed / duration : 0 }
}

public enum Audio8TTSError: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(missing: [String], unused: [String])
    case emptyGeneration

    public var description: String {
        switch self {
        case .missingFile(let path): return "Audio8TTS: missing file \(path)"
        case .keyContract(let missing, let unused):
            return "Audio8TTS key contract violated — missing \(missing.count) \(missing.prefix(5)), unused \(unused.count) \(unused.prefix(5))"
        case .emptyGeneration: return "Audio8TTS: model generated no audio frames"
        }
    }
}

public final class Audio8TTS: @unchecked Sendable {
    public let config: ArkttsConfig
    public let lm: ArkttsModel
    public let codec: ArkttsCodec
    public let directory: URL
    private var tokenizer: Tokenizer?

    init(config: ArkttsConfig, lm: ArkttsModel, codec: ArkttsCodec, directory: URL) {
        self.config = config
        self.lm = lm
        self.codec = codec
        self.directory = directory
    }

    /// Load from a directory holding the published mlx-community layout
    /// (config.json, model.safetensors, codec.safetensors, tokenizer.json).
    /// Weight IO rides the CPU stream (Metal-watchdog rule); the caller runs
    /// forwards on whatever stream is current.
    public static func load(directory: URL, lmDtype: DType = .bfloat16) throws -> Audio8TTS {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Audio8TTSError.missingFile(configURL.path)
        }
        let config = try ArkttsConfig.load(from: configURL)

        let (lmWeights, codecWeights) = try Device.withDefaultDevice(.cpu) {
            () throws -> ([String: MLXArray], [String: MLXArray]) in
            let lmRaw = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
            let codecRaw = try loadArrays(url: directory.appendingPathComponent("codec.safetensors"))
            var lm = [String: MLXArray]()
            for (key, value) in lmRaw where key.hasPrefix("model.") {
                lm[String(key.dropFirst("model.".count))] = value.asType(lmDtype)
            }
            var codec = [String: MLXArray]()
            for (key, value) in codecRaw where key.hasPrefix("codec.") {
                codec[String(key.dropFirst("codec.".count))] = value.asType(.float32)
            }
            eval(Array(lm.values) + Array(codec.values))
            return (lm, codec)
        }

        let lm = ArkttsModel(config: config)
        // S0: refuse partial loads — module keys must equal file keys exactly.
        let expected = Set(lm.parameters().flattened().map(\.0))
        let onDisk = Set(lmWeights.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw Audio8TTSError.keyContract(missing: missing, unused: unused)
        }
        try lm.update(
            parameters: ModuleParameters.unflattened(lmWeights.map { ($0.key, $0.value) }),
            verify: .all)
        // C14/INF: MLXNN.Module.training defaults to TRUE. Nothing in this graph branches on it
        // today (Linear/Embedding + a hand-rolled RMSNorm — no BatchNorm, no Dropout), but the
        // single construction choke point is the right place to pin inference mode so a future
        // layer addition cannot silently inherit training semantics.
        lm.train(false)
        eval(lm.parameters())

        let codec = ArkttsCodec(weights: codecWeights, config: config)
        return Audio8TTS(config: config, lm: lm, codec: codec, directory: directory)
    }

    // MARK: prompt building (mirrors processing_arktts.py)

    func loadTokenizer() async throws -> Tokenizer {
        if let tokenizer { return tokenizer }
        let loaded = try await AutoTokenizer.from(modelFolder: directory)
        tokenizer = loaded
        return loaded
    }

    static func cleanText(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    static func formatReferenceText(_ text: String) -> String {
        let cleaned = cleanText(text)
        if cleaned.range(of: #"<\|speaker:\d+\|>"#, options: .regularExpression) != nil {
            return cleaned
        }
        return "<|speaker:0|>\(cleaned)"
    }

    /// Returns (prefix, suffix) token rows for the prompt.
    public func promptSegments(
        text: String, referenceText: String?, hasReference: Bool
    ) async throws -> ([Int32], [Int32]) {
        let tokenizer = try await loadTokenizer()
        func encodeParts(_ parts: [String]) -> [Int32] {
            parts.flatMap { tokenizer.encode(text: $0, addSpecialTokens: false).map(Int32.init) }
        }
        let target = Self.cleanText(text)
        precondition(!target.isEmpty, "text must not be empty")
        if !hasReference {
            let full = encodeParts([
                "<|im_start|>system\n",
                "convert the provided text to speech",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ])
            return (full, [])
        }
        guard let referenceText, !referenceText.isEmpty else {
            preconditionFailure("reference_text is required when a reference voice is provided")
        }
        let prefix = encodeParts([
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            Self.formatReferenceText(referenceText),
            "\n\nSpeech:\n",
        ])
        let suffix = encodeParts([
            "<|im_end|>\n",
            "<|im_start|>user\n",
            target,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ])
        return (prefix, suffix)
    }

    // MARK: generation

    /// Encodes a reference clip to codec codes. Split out so callers doing long-form synthesis
    /// can encode ONCE and reuse across many `generate(...)` calls — the codec encoder is the
    /// expensive part of conditioning.
    /// - Parameter audio: mono fp32 samples at 44.1 kHz (resampling is the caller's concern).
    public func encodeReference(_ audio: MLXArray) -> (codes: MLXArray, length: Int) {
        let (codes, length) = codec.encode(
            audio.expandedDimensions(axis: 0), sampleCount: audio.shape[0])
        eval(codes)
        return (codes[0], length)
    }

    /// referenceAudio: mono fp32 samples at 44.1 kHz (resampling is the caller's concern).
    public func generate(
        text: String,
        referenceAudio: MLXArray? = nil,
        referenceText: String? = nil,
        params: SamplingParams = SamplingParams(),
        onFrame: ((Int) throws -> Void)? = nil
    ) async throws -> Audio8GenerationResult {
        let reference = referenceAudio.map { encodeReference($0) }
        return try await generate(
            text: text, referenceCodes: reference?.codes,
            referenceLength: reference?.length ?? 0, referenceText: referenceText,
            params: params, onFrame: onFrame)
    }

    /// Generation from PRE-ENCODED reference codes (see `encodeReference`).
    public func generate(
        text: String,
        referenceCodes: MLXArray?,
        referenceLength: Int,
        referenceText: String? = nil,
        params: SamplingParams = SamplingParams(),
        onFrame: ((Int) throws -> Void)? = nil
    ) async throws -> Audio8GenerationResult {
        let start = Date()
        let (prefix, suffix) = try await promptSegments(
            text: text, referenceText: referenceText, hasReference: referenceCodes != nil)

        let codes = try lm.generateCodes(
            prefix: prefix, suffix: suffix,
            referenceCodes: referenceCodes, referenceLength: referenceLength,
            params: params, onFrame: onFrame)
        eval(codes)
        let frames = codes.shape[2]
        guard frames > 0 else { throw Audio8TTSError.emptyGeneration }
        let waveform = codec.decode(codes)[0].asType(.float32)
        eval(waveform)
        return Audio8GenerationResult(
            audio: waveform, frames: frames, sampleRate: ArkttsCodec.sampleRate,
            elapsed: Date().timeIntervalSince(start))
    }
}
