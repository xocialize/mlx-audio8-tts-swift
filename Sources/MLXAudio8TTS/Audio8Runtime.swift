import Audio8TTSCore
import Foundation
import MLX
import MLXNN
import MLXToolKit

/// What the runtime needs from a package configuration to find its weights.
///
/// Both Audio8 checkpoints resolve weights identically — the only differences between them
/// are the default repo and the manifest — so this is the whole seam.
protocol Audio8WeightResolving: PackageConfiguration {
    var modelsRootDirectory: URL? { get }
    func missingWeightSources(storeRoot: URL?) -> [WeightSource]
    /// The weight directory after store resolution; nil when no explicit directory is set and
    /// no models root is available.
    func resolvedModelDirectory(storeRoot: URL?) -> URL?
}

/// The Audio8 inference runtime, shared by every checkpoint's `ModelPackage`.
///
/// The two published Audio8 checkpoints (0.6b `arktts`, 0.1b `falcon_h1`) differ ONLY in their
/// manifest — licence, footprints, provenance, surface summary — and in which repo their
/// configuration defaults to. The conditioning, the AR rollout, the cancellation cadence, the
/// metaData plane and the streaming decode are identical, and `Audio8TTS` already dispatches
/// the architectural difference internally from `config.json`.
///
/// So the packages are thin and this is where the behaviour lives. The alternative — a second
/// package duplicating ~200 lines of inference glue — is the shape that drifts: the failure
/// mode this file exists to prevent is a streaming twin quietly conditioning differently from
/// its batch sibling, and that risk doubles with every copy of the glue.
@InferenceActor
final class Audio8Runtime<C: Audio8WeightResolving> {
    let configuration: C
    private var model: Audio8TTS?

    /// C14/INF seam: the module graphs this package holds, exposed to the test target via
    /// `@testable`; the `InferenceModeInspectable` conformance lives there so the shipping
    /// target takes no dependency on the conformance library. `nil` before `load()`, so an
    /// unloaded package reports an empty graph — which INF-1 fails, by design.
    ///
    /// Audio8 carries NEITHER BatchNorm nor Dropout: the LM is Linear/Embedding plus a
    /// hand-rolled RMSNorm, and the codec is a functional port with no `Module` at all. The
    /// `train(false)` at `Audio8TTS.load`'s choke point is therefore hygiene rather than
    /// load-bearing today — it exists so a future layer cannot silently inherit training
    /// semantics. The test suite asserts that "no training-sensitive module" claim directly.
    var inferenceModeGraphs: [String: MLXNN.Module?] { ["lm": model?.lm] }
    /// Reference-conditioning reuse (the IndexTTS2/Qwen3/Gepard E1 pattern): long-form synthesis
    /// sends the SAME reference for every line, and encoding it re-runs the whole codec encoder.
    /// Memoized by reference-clip bytes. Safe to hold: `@InferenceActor` serializes `run()`, and
    /// the codes are read-only once built.
    private var cachedReference: (key: Int, codes: MLXArray, length: Int)?

    nonisolated init(_ configuration: C) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    func load() async throws {
        guard model == nil else { return }

        // Materialization is ENGINE-EXECUTED (contract 1.24): the engine downloads the declared
        // missing sources into the store BEFORE load(). This guard is the offline backstop only —
        // reaching it means no engine materialization ran (no store set, or a non-engine caller).
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw Audio8PackageError.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()

        guard let modelDir = configuration.resolvedModelDirectory(storeRoot: storeRoot) else {
            throw Audio8PackageError.missingWeights("unresolved weight directory (no store root)")
        }
        // Heavy: pages the bf16 LM + the fp32 codec. Weight IO rides the CPU stream inside
        // `Audio8TTS.load` (Metal-watchdog rule); the tokenizer loads off the same directory.
        model = try Audio8TTS.load(directory: modelDir)
    }

    func unload() async {
        model = nil
        cachedReference = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the AR rollout bails per generated frame via the
        // `onFrame` predicate, and the CancellationError is rethrown UNCHANGED so the engine can
        // classify user-cancel vs governor-preempt.
        try Task.checkCancellation()
        let (model, conditioning, params) = try prepared(for: request)
        try Task.checkCancellation()

        // The DualAR rollout, then the codec decode. Cancellation + progress ride the per-frame
        // seam; `Task.checkCancellation` throws CancellationError, which propagates UNCHANGED
        // through the core's rethrowing `onFrame` so the engine can classify it (CAN-2).
        let result = try await model.generate(
            text: conditioning.text,
            referenceCodes: conditioning.codes,
            referenceLength: conditioning.length,
            referenceText: conditioning.transcript,
            params: params,
            onFrame: { count in
                RunProgress.report(.generate, step: count)
                try Task.checkCancellation()
            })
        try Task.checkCancellation()

        RunProgress.report(.decode)
        let samples: [Float] = result.audio.asArray(Float.self)
        let wav = AudioSupport.encodeWAV16(samples: samples, sampleRate: result.sampleRate)
        return TTSResponse(audio: Audio(
            format: .wav, data: wav, sampleRate: result.sampleRate, channels: 1))
    }

    // MARK: - Shared conditioning (run + runStream)

    struct Conditioning {
        let text: String
        let codes: MLXArray?
        let length: Int
        let transcript: String?
    }

    /// Voice guard → memoized reference codes → metaData-plane sampling options. Shared by
    /// `run()` and `runStream()` so the two paths cannot drift apart — the failure mode where a
    /// streaming twin quietly conditions differently from its batch sibling.
    func prepared(for request: any CapabilityRequest) throws
        -> (Audio8TTS, Conditioning, SamplingParams) {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Voice plane: zero-shot cloning (audio + transcript) or the model's default voice.
        // The codec encode is memoized per clip — long-form synthesis sends the same reference
        // for every sentence, and re-encoding it dominates conditioning cost.
        var referenceCodes: MLXArray?
        var referenceLength = 0
        var referenceText: String?
        switch tts.voice.selection {
        case .referenceAudio(let clip):
            guard let transcript = tts.referenceTranscript, !transcript.isEmpty else {
                throw PackageError.unsupportedRequestFeature(
                    "referenceTranscript — Audio8-TTS is an ICL cloner: the reference clip must "
                    + "be paired with a transcript of what it says")
            }
            let key = Self.referenceKey(clip.data)
            if let cached = cachedReference, cached.key == key {
                (referenceCodes, referenceLength) = (cached.codes, cached.length)
            } else {
                RunProgress.report(.encode)
                let (samples, sourceRate) = try AudioSupport.decodeToMono(clip)
                let resampled = SincResampler.resample(
                    audio: samples, from: sourceRate, to: ArkttsCodec.sampleRate)
                let encoded = model.encodeReference(MLXArray(resampled))
                cachedReference = (key, encoded.codes, encoded.length)
                (referenceCodes, referenceLength) = (encoded.codes, encoded.length)
            }
            referenceText = transcript
        case .auto:
            break   // no reference — the model synthesizes with its own default voice
        case .named(let name):
            throw PackageError.unsupportedRequestFeature(
                "voice.named(\"\(name)\") — Audio8-TTS has no preset voice ids; use "
                + "voice.referenceAudio with a referenceTranscript, or voice.auto")
        }

        // metaData plane.
        var params = SamplingParams()
        if let value = tts.metaData.intValue("maxFrames") { params.maxNewTokens = max(1, value) }
        if let value = tts.metaData.doubleValue("temperature") {
            params.temperature = Float(max(value, 1e-5))
        }
        if let value = tts.metaData.doubleValue("topP") {
            params.topP = Float(min(max(value, 1e-3), 1.0))
        }
        if let value = tts.metaData.intValue("topK") { params.topK = max(1, value) }
        if case .bool(true)? = tts.metaData["greedy"] { params.doSample = false }
        if let seed = tts.metaData.intValue("seed") {
            params.seed = UInt64(bitPattern: Int64(seed))
        }

        return (model, Conditioning(text: tts.text, codes: referenceCodes,
                                    length: referenceLength, transcript: referenceText), params)
    }

    // MARK: - Streaming (StreamEmitting)

    /// Streaming twin of `run()`: same conditioning, same AR rollout, but the codec decode is
    /// windowed and incremental, so audio leaves before the utterance finishes and peak
    /// activation is bounded by the chunk rather than by utterance length.
    ///
    /// Exactness is not assumed: the decoder stack below `post_module` is strictly causal, and
    /// the `--s5` gate proves the chunked decode is bit-identical (max_abs exactly 0) to the
    /// whole-utterance decode at the default chunk size. Returns the same aggregated response
    /// `run()` would have produced.
    ///
    /// `metaData.streamChunkFrames` (int) tunes the cadence; it is clamped to the codec's
    /// measured exactness floor, below which small-input conv kernels break bit-identity.
    func runStream(_ request: any CapabilityRequest,
                          emit: @escaping @Sendable (TTSStreamChunk) -> Void)
        async throws -> any CapabilityResponse {
        try Task.checkCancellation()   // entry checkpoint precedes the first emit
        let (model, conditioning, params) = try prepared(for: request)
        try Task.checkCancellation()

        let tts = request as? TTSRequest
        let requested = tts?.metaData.intValue("streamChunkFrames") ?? 64
        var index = 0
        var all: [Float] = []

        let result = try await model.generateStreaming(
            text: conditioning.text,
            referenceCodes: conditioning.codes,
            referenceLength: conditioning.length,
            referenceText: conditioning.transcript,
            params: params,
            chunkFrames: requested,
            onFrame: { count in
                RunProgress.report(.generate, step: count)
                try Task.checkCancellation()
            },
            onChunk: { samples, isFinal in
                emit(TTSStreamChunk(samples: samples,
                                    sampleRate: ArkttsCodec.sampleRate,
                                    index: index, isFinal: isFinal))
                index += 1
                all.append(contentsOf: samples)
            })
        try Task.checkCancellation()

        RunProgress.report(.decode)
        _ = result
        let wav = AudioSupport.encodeWAV16(samples: all, sampleRate: ArkttsCodec.sampleRate)
        return TTSResponse(audio: Audio(
            format: .wav, data: wav, sampleRate: ArkttsCodec.sampleRate, channels: 1))
    }

    /// In-memory cache key for a reference clip (Hasher is per-process seeded, which is all we
    /// need — reuse happens within one long-form run).
    nonisolated static func referenceKey(_ data: Data) -> Int {
        var hasher = Hasher()
        hasher.combine(data)
        return hasher.finalize()
    }
}
