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

    private let runtime: Audio8Runtime<Audio8Configuration>

    /// C14/INF seam, forwarded from the runtime — see `Audio8Runtime.inferenceModeGraphs`.
    var inferenceModeGraphs: [String: MLXNN.Module?] { runtime.inferenceModeGraphs }

    public nonisolated init(configuration: Configuration) {
        self.runtime = Audio8Runtime(configuration)
    }

    // MARK: - Lifecycle + run, delegated
    //
    // Everything below is checkpoint-independent and lives in `Audio8Runtime`, shared with
    // `Audio8MiniPackage`. Only the manifest above distinguishes the two.

    public func load() async throws { try await runtime.load() }
    public func unload() async { await runtime.unload() }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await runtime.run(request)
    }

    public func runStream(_ request: any CapabilityRequest,
                          emit: @escaping @Sendable (TTSStreamChunk) -> Void)
        async throws -> any CapabilityResponse {
        try await runtime.runStream(request, emit: emit)
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
