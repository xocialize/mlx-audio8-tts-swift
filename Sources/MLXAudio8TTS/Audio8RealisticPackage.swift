import Audio8TTSCore
import Foundation
import MLX
import MLXNN
import MLXToolKit

// MARK: - Configuration

/// Init-time configuration for `Audio8RealisticPackage` (C9).
///
/// Its own type, for the reason `Audio8MiniConfiguration` is: the DEFAULT repo. The Realistic
/// and Preview checkpoints share every tensor name and shape, so a Preview configuration pointed
/// at the wrong default would load and run without complaint — and that silence is the hazard.
///
/// `quant` is `.bf16` as-shipped: the same 601M-param LM shape as the Preview (~1.2 GB bf16) and
/// the same fp32 codec, which is byte-identical to the Preview bundle's `codec.safetensors`.
public struct Audio8RealisticConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The MLX conversion (LM + codec + tokenizer + config), published by us.
    public var repo: String
    /// Pinned revision; nil = main.
    public var revision: String?
    /// Quant tier: `.bf16` as-shipped.
    public var quant: Quant
    /// Explicit checkpoint directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Audio8-TTS-Realistic-0.6b-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case repo, revision, quant }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
    }
}

extension Audio8RealisticConfiguration: WeightSourcing {
    static let mainFiles = [
        "model.safetensors", "codec.safetensors", "config.json",
        "tokenizer.json", "tokenizer_config.json",
    ]
    /// Representative file for the missing-probe (the codec — the largest single artifact).
    static let mainProbeFile = "codec.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, revision: revision, matching: Self.mainFiles)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        if let dir = modelDirectory,
           fm.fileExists(atPath: dir.appending(path: Self.mainProbeFile).path) {
            return []
        }
        guard let dir = ModelStore(root: storeRoot).directory(for: repo),
              fm.fileExists(atPath: dir.appending(path: Self.mainProbeFile).path)
        else { return weightSources }
        return []
    }

    public func resolved(storeRoot: URL?) -> Audio8RealisticConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

extension Audio8RealisticConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return [
            dir.appending(path: "model.safetensors"),
            dir.appending(path: Self.mainProbeFile),
        ]
    }
}

extension Audio8RealisticConfiguration: Audio8WeightResolving {
    func resolvedModelDirectory(storeRoot: URL?) -> URL? {
        resolved(storeRoot: storeRoot).modelDirectory
    }
}

// MARK: - Package

/// Audio8-TTS-Realistic-Air on the canonical `tts` surface — **the recommended 0.6b
/// checkpoint**, a weights-only successor to `Audio8Package`'s Preview-0.6b.
///
/// Upstream renamed the repo from `Edge0/Audio8-TTS-Realistic-0.6b` to
/// `Edge0/Audio8-TTS-Realistic-Air` on 2026-11-02. The weights did not change: the pinned
/// revision resolves under the new name, and the commits after it only edit the model card
/// (AB-R-0719). "Air" is a release name, not a smaller model.
///
/// Upstream (Edge0) trained it on top of the Preview with SFT, then GRPO against WER and
/// speaker-similarity rewards, then GRPO against a realism critic. Code, codec, tokenizer,
/// processor and the 226-tensor LM layout are byte-identical to the Preview's; only the LM
/// tensor VALUES differ. So everything behavioural is shared through `Audio8Runtime`, and this
/// manifest is the only thing that distinguishes the two packages.
///
/// **What changes against the Preview, measured** (E28 V0, 608 paired takes through the
/// package; `WIP/audio8-tts/realistic-0.6b/MEASUREMENTS.md`):
///
/// - **Tied on the core languages** (en, zh, de, es, fr, ja): macro error 0.0126 vs 0.0122,
///   speaker similarity 0.7211 vs 0.7212, no catastrophic takes on either.
/// - **More pauses.** Takes run ×1.126 longer, and all of it is silence: pause time per line
///   0.62 → 1.52 s while articulation is slightly *faster* (×0.97). Callers fitting speech to a
///   time slot should budget for it; no request knob turns it down.
/// - **Many more languages.** Malay, Turkish, Portuguese, Indonesian, Finnish and Russian score
///   1–7 % error where the Preview scored 11–75 %. Swedish, Norwegian and Hindi are usable but
///   uneven. Arabic, Vietnamese and Thai are NOT declared (Thai loops on every take).
///
/// `metaData` keys are identical to `Audio8Package`'s: `maxFrames`, `temperature`, `topP`,
/// `topK`, `greedy`, `seed`, `streamChunkFrames`.
@InferenceActor
public final class Audio8RealisticPackage: ModelPackage, StreamEmitting {
    public typealias Configuration = Audio8RealisticConfiguration

    /// Split footprints — MEASURED through this package's own lifecycle (`audio8-gates
    /// --validate` and `--streamenv` against the release repo), never estimated. The gates
    /// print MiB; these are bytes, margins applied after converting (AB-L-0116).
    ///
    ///   • resident floor (MLX-active, post-load, pre-run): **2434 MiB** → ×1.07 → 2.75e9 B.
    ///   • activation transient (peak − resident): stream flat at 3482 MiB from 79 to 527
    ///     frames; batch at most **3658 MiB** (one 79-frame window) → ×1.13 → 4.35e9 B.
    ///
    /// Identical to the Preview's to the megabyte, as it must be — same tensor shapes, same
    /// dtypes, same windowed decode. Measured anyway, because "the graph is unchanged" is the
    /// kind of argument a declaration should not rest on — and measuring it is what found the
    /// batch-window overshoot both checkpoints carried until v0.4.0 (see `decodeStreaming`).
    nonisolated static let bf16ResidentBytes: UInt64 = 2_750_000_000
    nonisolated static let peakActivationBytes: UInt64 = 4_350_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: ONE weight layer, Apache-2.0 all the way down — the LM is Apache-2.0 (Edge0),
            // fine-tuned from the Apache-2.0 Preview, whose slow AR is Qwen2.5-0.5B (Apache-2.0);
            // the codec is the Preview's, byte for byte. Admits under `.permissiveOnly`.
            // C8: the port code (this repo) is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "Edge0/Audio8-TTS-Realistic-Air",
                revision: "925bbd4c2d8360af85f2594bf0205173fda853ba", tier: 2),
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
                // EQUAL to the Preview's, not higher: cross-lingual SIM measured 0.7211 against
                // 0.7212. The card's +0.02 SIM came from same-language references and did not
                // reproduce on ours, so claiming more would be the card talking, not the gate.
                SpecialtyWeight(.voiceClone, strength: 1.0),
                // A property of the shared codec decode, unchanged.
                SpecialtyWeight(.realtimeStreaming, strength: 0.7),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "audio8-tts-realistic",
                    summary: "Audio8-TTS-Realistic-0.6b multilingual zero-shot voice-cloning "
                        + "TTS (.wav, 44.1 kHz mono) — the RL-tuned successor to audio8-tts, "
                        + "same architecture and codec, at 21.5 frames/s. Clones from "
                        + "voice.referenceAudio + referenceTranscript (the transcript must "
                        + "match the clip); .auto uses the model's default voice. 17 languages: "
                        + "Cantonese, Chinese, Dutch, English, Finnish, French, German, "
                        + "Indonesian, Italian, Japanese, Korean, Malay, Polish, Portuguese, "
                        + "Russian, Spanish, Turkish; experimental (usable, uneven): Hindi, "
                        + "Norwegian, Swedish. Not supported: Arabic, Thai, Vietnamese. Pauses "
                        + "more than audio8-tts (takes ~13% longer, all of it silence); prefer "
                        + "audio8-tts when speech must fit a tight time slot. metaData: "
                        + "temperature/topP/topK/seed/greedy/maxFrames/streamChunkFrames. "
                        + "Streams PCM chunks via the same bit-identical windowed decode.",
                    modes: [.neutral],
                    streaming: .audioChunk
                )
            ]
        )
    }

    private let runtime: Audio8Runtime<Audio8RealisticConfiguration>

    /// C14/INF seam, forwarded from the runtime — see `Audio8Runtime.inferenceModeGraphs`.
    var inferenceModeGraphs: [String: MLXNN.Module?] { runtime.inferenceModeGraphs }

    public nonisolated init(configuration: Configuration) {
        self.runtime = Audio8Runtime(configuration)
    }

    // MARK: - Lifecycle + run, delegated (see Audio8Runtime)

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
