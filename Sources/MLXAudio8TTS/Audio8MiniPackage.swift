import Audio8TTSCore
import Foundation
import MLX
import MLXNN
import MLXToolKit

// MARK: - Licence

extension SPDXLicense {
    /// Audio8's Community License v1.0, carried by the 0.1b weights. Reviewed against the full
    /// text shipped in the model repo:
    ///
    /// - **§2.1** Non-Commercial Use — reproduce, modify and redistribute, freely.
    /// - **§2.2** Commercial Use **permitted** below **US$2,000,000** consolidated Annual Revenue.
    /// - **§2.3** At or above that threshold, Commercial Use needs a separate written licence.
    /// - **§3** Attribution on redistribution. No non-compete, no derivative restriction, no
    ///   eval-only clause.
    ///
    /// Same shape as MLXToolKit's allowlisted `ltx2Community` (source-available with a revenue
    /// gate) and without that licence's §A.20 non-compete — but at a **5x lower threshold**,
    /// US$2M against US$10M, which is low enough to bind a real business rather than only a
    /// large one. Worth stating as a number rather than as "revenue-capped, like LTX-2".
    ///
    /// Defined HERE rather than in MLXToolKit on purpose. Allowlisting changes what
    /// `.permissiveOnly` admits fleet-wide and wants a contract bump plus a tagged engine
    /// release — not something an audio port should land as a side effect. The engine-side
    /// change is prepared on `mlx-engine-swift@audio8-community-license`; until it merges this
    /// declaration is accurate but not on `permissiveAllowlist`, so the engine records a
    /// `LicenseAdvisory` instead of admitting it silently. Enforcement defaults to `.advisory`
    /// (contract 1.28.0), so that classifies rather than blocks — and a revenue cap is exactly
    /// the kind of term a consuming app should be showing its users anyway. Delete this
    /// extension when the engine constant lands.
    public static let audio8Community: SPDXLicense = "LicenseRef-Audio8-Community"
}

// MARK: - Configuration

/// Init-time configuration for `Audio8MiniPackage` (C9).
///
/// A distinct type from `Audio8Configuration` for one reason: the DEFAULT repo. Sharing the
/// 0.6b's configuration would leave a default pointing at the wrong checkpoint, and a package
/// that silently loads a different model than its manifest declares is a worse failure than a
/// little duplication.
///
/// `quant` is `.bf16` as-shipped: the ~170M-param LM binds at native bf16 (~340 MB) and the
/// codec stays fp32 — the same precision-sensitive path as the 0.6b (Snake activations,
/// cosine-normalized VQ lookups), and literally the same tensors.
public struct Audio8MiniConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
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
        repo: String = "mlx-community/Audio8-TTS-Preview-0.1b-bf16",
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

extension Audio8MiniConfiguration: WeightSourcing {
    static let mainFiles = [
        "model.safetensors", "codec.safetensors", "config.json",
        "tokenizer.json", "tokenizer_config.json",
    ]
    /// Representative file for the missing-probe — the codec, still the largest single artifact
    /// here by a wide margin (1.35 GB against the LM's 340 MB).
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

    public func resolved(storeRoot: URL?) -> Audio8MiniConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

extension Audio8MiniConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return [
            dir.appending(path: "model.safetensors"),
            dir.appending(path: Self.mainProbeFile),
        ]
    }
}

extension Audio8MiniConfiguration: Audio8WeightResolving {
    func resolvedModelDirectory(storeRoot: URL?) -> URL? {
        resolved(storeRoot: storeRoot).modelDirectory
    }
}

// MARK: - Package

/// Audio8-TTS-Preview-0.1b on the canonical `tts` surface — the compact sibling of
/// `Audio8Package`. ~170M-parameter LM with a Falcon-H1 hybrid slow stack (Mamba-2 + attention
/// per layer), the same 4-layer fast AR, and the **same 44.1 kHz codec, byte for byte**.
///
/// Everything behavioural is shared with the 0.6b through `Audio8Runtime`; `Audio8TTS`
/// dispatches the architectural difference internally from `slow_backbone` in `config.json`.
/// Only this manifest distinguishes the two packages.
///
/// **Choose this over `Audio8Package` for footprint, not for quality.** It is 1.7 GB against
/// 2.6 GB on disk and roughly 800 MB lighter resident, but speaker similarity — the metric
/// that matters for zero-shot cloning — is measurably worse (upstream Seed-TTS SIM 56.7 EN /
/// 68.2 ZH against the 0.6b's 63.2 / 73.1), and it supports 8 languages rather than 11. The
/// peak-activation saving is small because the shared codec dominates that axis.
///
/// `metaData` keys are identical to `Audio8Package`'s: `maxFrames`, `temperature`, `topP`,
/// `topK`, `greedy`, `seed`, `streamChunkFrames`.
@InferenceActor
public final class Audio8MiniPackage: ModelPackage, StreamEmitting {
    public typealias Configuration = Audio8MiniConfiguration

    /// Split footprints — MEASURED through this package's own lifecycle (`audio8-gates
    /// --validate` and `--streamenv` against the release repo), never estimated.
    ///
    ///   • resident floor (MLX-active, post-load, pre-run): **1611 MB** → declared 1.81 GB.
    ///   • activation transient (peak − resident): **3711 MB** max observed → declared 4.40 GB.
    ///
    /// The envelope is FLAT in utterance length, the same property the 0.6b gained when the
    /// windowed decode landed — and for the same reason, since both share that decode path:
    ///
    ///     frames   batch peak   stream peak
    ///         64      3457 MB       3474 MB
    ///        128      3457 MB       3474 MB
    ///        234      3458 MB       3474 MB
    ///
    /// **Honest limit on that evidence:** the sweep requests 64/128/256/512 but this checkpoint
    /// hit EOS at 234 frames on the sweep text, so 512 was never reached. Flatness here rests on
    /// 64→234 measured plus the shared, length-independent decode mechanism the 0.6b verified to
    /// 1035 frames — not on a 512-frame measurement of this checkpoint. The 0.6b's declaration
    /// was revised upward three times (5.00 → 7.20 → 9.50 GB) precisely because a sample never
    /// reached the cap, so the distinction is worth keeping visible rather than rounding off.
    ///
    /// Note the resident saving is smaller than the 3.5x LM shrink suggests: the codec is
    /// unchanged and is ~80% of what remains.
    nonisolated static let bf16ResidentBytes: UInt64 = 1_810_000_000
    nonisolated static let peakActivationBytes: UInt64 = 4_400_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weight layer is genuinely TWO licences, and the manifest can carry one —
            // so it declares the binding one. The LM is the Audio8 Community Licence
            // (revenue-capped at US$2M). The bundled codec is Apache-2.0: upstream ships a
            // byte-identical codec.pth in the Apache-2.0 0.6b repo (same LFS oid; verified
            // bit-identical across encoder, quantizer decode and full decode), and our tensors
            // are converted from that repo. Declaring the stricter of the two is the honest
            // reduction — a consumer cleared for the LM is cleared for both.
            // C8: the port code (this repo) is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .audio8Community, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "Audio8/Audio8-TTS-Preview-0.1b",
                revision: "b476f0208438dfa791abee44d11029f055aeae04", tier: 2),
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
                // Deliberately BELOW the 0.6b's 1.0. Strength is a selection signal, and this
                // checkpoint clones measurably less faithfully (Seed-TTS SIM 56.7/68.2 against
                // 63.2/73.1). Copying the sibling's weight would tell the selector the two are
                // interchangeable for the one job they are picked for.
                SpecialtyWeight(.voiceClone, strength: 0.8),
                // Streaming is a property of the shared codec decode, not of the LM, so this
                // claim carries over unchanged — and the AR half is faster here.
                SpecialtyWeight(.realtimeStreaming, strength: 0.7),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "audio8-tts-mini",
                    summary: "Audio8-TTS-Preview-0.1b compact multilingual zero-shot "
                        + "voice-cloning TTS (.wav, 44.1 kHz mono). ~170M-parameter LM with a "
                        + "Falcon-H1 hybrid slow stack; same DualAR shape and same 44.1 kHz "
                        + "codec as audio8-tts, at 21.5 frames/s. Clones from "
                        + "voice.referenceAudio + referenceTranscript (the transcript must "
                        + "match the clip); .auto uses the model's default voice. 8 languages: "
                        + "Chinese and English are primary; German, Spanish, French, Italian, "
                        + "Japanese and Korean are weaker and more variable than in audio8-tts. "
                        + "Prefer audio8-tts when speaker similarity matters; prefer this when "
                        + "footprint does (1.7 GB vs 2.6 GB on disk). metaData: "
                        + "temperature/topP/topK/seed/greedy/maxFrames/streamChunkFrames. "
                        + "Streams PCM chunks via the same bit-identical windowed decode.",
                    modes: [.neutral],
                    streaming: .audioChunk
                )
            ]
        )
    }

    private let runtime: Audio8Runtime<Audio8MiniConfiguration>

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
