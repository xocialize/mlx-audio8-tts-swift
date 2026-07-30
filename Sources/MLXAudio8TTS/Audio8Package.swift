import Audio8TTSCore
import Foundation
import MLX
import MLXNN
import MLXToolKit

/// Audio8-TTS-Preview-0.6b on the canonical `tts` surface: multilingual (11 languages)
/// zero-shot voice cloning from a reference clip + its transcript. Returns the canonical
/// `Audio` (.wav, 44.1 kHz mono).
///
/// Engine-owned lifecycle (C13): the engine constructs from an `Audio8Configuration`, pages
/// weights in with `load()` (auto-materializing the declared source under the engine's models
/// root when set), drives `run(_:)`, and reclaims with `unload()`.
///
/// Voice: `.referenceAudio` + `referenceTranscript` (ICL-grade cloning — the transcript must
/// match what is spoken in the clip; the model conditions on the pair), or `.auto` for the
/// model's own default voice. `.named` has no meaning here and rejects legibly.
///
/// `metaData` keys (package-specific, C5):
/// - `maxFrames` (int, default 512, capped by `max_seq_len − prompt`): generation cap. One
///   frame ≈ 46 ms of audio (2048 samples @ 44.1 kHz), so 512 ≈ 23.8 s.
/// - `temperature` (double, default 0.7), `topP` (double, default 0.9), `topK` (int,
///   default 50): the reference sampler's knobs, applied in the reference's order
///   (filter → temperature → exponential race).
/// - `greedy` (bool, default false): deterministic argmax decoding. This is the
///   configuration the S2 gate proves token-exact against the PyTorch reference.
/// - `seed` (int): reproducible sampling. Unlike the fleet's greedy-only packages this is
///   NOT a no-op — the default path samples.
@InferenceActor
public final class Audio8Package: ModelPackage, StreamEmitting {
    public typealias Configuration = Audio8Configuration

    /// Split footprints — MEASURED. MEASUREMENTS.md carries every pass.
    ///   • resident floor (MLX-active, post-load, pre-run): 2434 MB → declared 2.60 GB.
    ///   • activation: **FLAT at 3482 MB**, independent of utterance length → declared 4.20 GB.
    ///
    /// The activation envelope stopped being a function of the input in v0.2.0, when the
    /// windowed decode landed and `generate` was routed through it. Before that the transient
    /// was linear in frames (`≈ 1824 + 14.2 × frames` MB) and every declaration under-shot it,
    /// three times running, because a sample never reaches the cap:
    ///
    ///     v0.1.0  5.00 GB   from one 9.2 s utterance
    ///     v0.1.1  7.20 GB   from a corpus sweep topping out at 15.7 s
    ///     v0.1.2  9.50 GB   from the fitted model at the default maxFrames cap
    ///     v0.2.1  4.20 GB   MEASURED FLAT — the length dependence is gone
    ///
    /// Verified across 64 → 224 frames in one sweep and against a 1035-frame run: 3482 MB every
    /// time. That is worth more than the 5.3 GB reduction itself — a declaration the governor
    /// reserves is only meaningful if the real envelope cannot exceed it, and this one now
    /// cannot. `maxFrames` no longer changes the memory story, only the duration.
    nonisolated static let bf16ResidentBytes: UInt64 = 2_600_000_000
    nonisolated static let peakActivationBytes: UInt64 = 4_200_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: ONE weight layer — the MLX conversion of an Apache-2.0 upstream (LM and codec
            // ship in the same repo under the same license), so the package admits under the
            // default `.permissiveOnly`. C8: the port code (this repo) is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "Audio8/Audio8-TTS-Preview-0.6b",
                revision: "1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17", tier: 2),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes,
                                   peakActivationBytes: peakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [
                // Zero-shot cloning is the selection axis. `.realtimeStreaming` is claimed only
                // now that the windowed decode exists — before it, whole-utterance decoding made
                // the claim false. NOT .emotionControl/.durationControl
                // (no native control plane). The multilingual reach — 11 languages, the real
                // differentiator vs the fleet's other cloners — has no registered Specialty
                // term; it is stated in the surface summary instead of inventing vocabulary.
                SpecialtyWeight(.voiceClone, strength: 1.0),
                SpecialtyWeight(.realtimeStreaming, strength: 0.7),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "audio8-tts",
                    summary: "Audio8-TTS-Preview-0.6b multilingual zero-shot voice-cloning TTS "
                        + "(.wav, 44.1 kHz mono). Clones from voice.referenceAudio + "
                        + "referenceTranscript (the transcript must match the clip); .auto uses "
                        + "the model's default voice. 11 languages: Cantonese, Chinese, Dutch, "
                        + "English, French, German, Italian, Japanese, Korean, Polish, Spanish. "
                        + "DualAR architecture at 21.5 frames/s; whole-utterance synthesis "
                        + "metaData: temperature/topP/topK/seed/greedy/maxFrames/streamChunkFrames. "
                        + "Streams PCM chunks: the codec decoder is strictly causal, so the "
                        + "windowed incremental decode is bit-identical to the whole-utterance "
                        + "one (gated) while bounding activation by the chunk.",
                    modes: [.neutral],
                    streaming: .audioChunk
                )
            ]
        )
    }

    private let configuration: Configuration
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

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
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

        guard let modelDir = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw Audio8PackageError.missingWeights("unresolved weight directory (no store root)")
        }
        // Heavy: pages the bf16 LM + the fp32 codec. Weight IO rides the CPU stream inside
        // `Audio8TTS.load` (Metal-watchdog rule); the tokenizer loads off the same directory.
        model = try Audio8TTS.load(directory: modelDir)
    }

    public func unload() async {
        model = nil
        cachedReference = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
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
    private func prepared(for request: any CapabilityRequest) throws
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
    public func runStream(_ request: any CapabilityRequest,
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

/// Wrapper-level errors (weight resolution). Runtime request errors use `PackageError`.
public enum Audio8PackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    public var description: String {
        switch self {
        case .missingWeights(let why): return "Audio8-TTS weights unavailable: \(why)"
        }
    }
}

extension MetaData {
    /// Convenience: read an int-valued metaData key.
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }

    /// Convenience: read a double-valued key, accepting ints (JSON 1 vs 1.0).
    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
}
