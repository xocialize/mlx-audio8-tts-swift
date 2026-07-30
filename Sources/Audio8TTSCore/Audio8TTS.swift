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

    /// Streaming twin of `generate`: identical AR rollout, but the codec decode is windowed
    /// and incremental, so each chunk's samples are handed to `onChunk` as soon as they exist
    /// and peak activation is bounded by the chunk instead of the utterance.
    ///
    /// Returns the same aggregated result `generate` would have produced — the streamed chunks
    /// concatenate to exactly that waveform (proven bit-identical by the `--s5` gate at the
    /// default chunk size).
    public func generateStreaming(
        text: String,
        referenceCodes: MLXArray?,
        referenceLength: Int,
        referenceText: String? = nil,
        params: SamplingParams = SamplingParams(),
        chunkFrames: Int? = nil,
        onFrame: ((Int) throws -> Void)? = nil,
        onChunk: @escaping ([Float], Bool) throws -> Void
    ) async throws -> Audio8GenerationResult {
        let start = Date()
        let (prefix, suffix) = try await promptSegments(
            text: text, referenceText: referenceText, hasReference: referenceCodes != nil)

        // INTERLEAVED: decode and emit while the rollout is still running. `post_module` is
        // causal, so running it over a PREFIX yields identical values for every position in that
        // prefix as running it over the whole sequence — which is what makes decoding ahead of
        // the final frame legitimate rather than approximate.
        var emitted: [MLXArray] = []
        var pieces: [MLXArray] = []
        var decodedUpTo = 0
        let chunk = max(codec.minimumExactChunkFrames, chunkFrames ?? 64)
        let context = codec.leftReceptiveFieldFrames

        func drain(force: Bool) throws {
            while emitted.count - decodedUpTo >= chunk || (force && decodedUpTo < emitted.count) {
                let end = force ? emitted.count : min(decodedUpTo + chunk, emitted.count)
                guard end > decodedUpTo else { break }
                // `post_module` runs over the FULL prefix, never a window. Two reasons it cannot
                // be windowed here, both of which produce plausible-but-wrong audio rather than
                // an error: its receptive field is 1016 frames (8 stacked 128-wide layers), and
                // its RoPE is absolute — a window starting at frame N would re-phase those
                // positions as if they were 0..k. Running it on the prefix is exact because the
                // module is causal, and cheap because it is a 1024-wide transformer, not the
                // conv stack. ONLY `decodeFromLatent` is windowed, at its 11-frame field.
                let prefix = concatenated(Array(emitted[0 ..< end]), axis: 2)
                let latentAll = codec.postModuleLatent(prefix)
                let windowStart = max(0, decodedUpTo - context)
                let decoded = codec.decodeFromLatent(latentAll[0..., windowStart ..< end])
                let drop = (decodedUpTo - windowStart) * ArkttsCodec.frameLength
                let mono = decoded[0, drop...].asType(.float32)
                eval(mono)
                let isFinal = force && end == emitted.count
                try onChunk(mono.asArray(Float.self), isFinal)
                pieces.append(mono)
                decodedUpTo = end
                if force { break }
            }
        }

        let codes = try lm.generateCodes(
            prefix: prefix, suffix: suffix,
            referenceCodes: referenceCodes, referenceLength: referenceLength,
            params: params,
            onFrame: onFrame,
            onFrameCodes: { frame in
                emitted.append(frame.expandedDimensions(axis: 2))
                try drain(force: false)
            })
        eval(codes)
        let frames = codes.shape[2]
        guard frames > 0 else { throw Audio8TTSError.emptyGeneration }
        try drain(force: true)

        let waveform = concatenated(pieces, axis: 0)
        eval(waveform)
        return Audio8GenerationResult(
            audio: waveform, frames: frames, sampleRate: ArkttsCodec.sampleRate,
            elapsed: Date().timeIntervalSince(start))
    }

    /// Generation from PRE-ENCODED reference codes (see `encodeReference`).
    ///
    /// Uses the windowed decode — bit-identical to a whole-utterance decode (`--s5`) but with a
    /// FLAT activation envelope instead of one linear in frames. It does NOT go through
    /// `generateStreaming`: that path re-runs `post_module` on a growing prefix for every chunk
    /// to get audio out early, which is quadratic work a batch caller has no use for. Here
    /// `post_module` runs exactly once and only the conv stack is windowed.
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

        var pieces: [MLXArray] = []
        try codec.decodeStreaming(codes) { samples, _ in
            let mono = samples[0].asType(.float32)
            eval(mono)
            pieces.append(mono)
        }
        let waveform = concatenated(pieces, axis: 0)
        eval(waveform)
        return Audio8GenerationResult(
            audio: waveform, frames: frames, sampleRate: ArkttsCodec.sampleRate,
            elapsed: Date().timeIntervalSince(start))
    }
}
