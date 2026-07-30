import Foundation
import MLXToolKit

/// Init-time configuration for `Audio8Package` (C9). Per-request text/voice/metaData ride the
/// canonical `TTSRequest`.
///
/// ONE weight source backs the loaded model: the MLX conversion publishes the DualAR LM
/// (`model.safetensors`, bf16), the 44.1 kHz codec (`codec.safetensors`, fp32, pre-folded and
/// already in MLX conv layout), `config.json`, and the Qwen `tokenizer.json` pair — all in one
/// Apache-2.0 repo, so the license gate is single-layer and admits under the default
/// `.permissiveOnly` policy.
///
/// `quant` is `.bf16` as-shipped: the 601M-param LM binds at native bf16 (~1.2 GB) and the codec
/// stays fp32 (the S1-validated path — it carries Snake activations and cosine-normalized VQ
/// lookups that are precision-sensitive). int8 for the LM Linears is a future tier.
public struct Audio8Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The MLX-converted Audio8-TTS repo (LM + codec + tokenizer + config).
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
        repo: String = "mlx-community/Audio8-TTS-Preview-0.6b-bf16",
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

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension Audio8Configuration: WeightSourcing {
    /// Everything the model needs: LM weights, codec weights, config, and the tokenizer pair.
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

    /// The configuration with a nil directory resolved to the store layout — what `load()` uses
    /// AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> Audio8Configuration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension Audio8Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved so auto-materialized (nil-dir) configs prewarm the downloaded layout on
        // later cold launches; missing paths are skipped (best-effort prewarmer).
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return [
            dir.appending(path: "model.safetensors"),
            dir.appending(path: Self.mainProbeFile),
        ]
    }
}
