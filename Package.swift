// swift-tools-version: 6.2
// mlx-audio8-tts-swift — Audio8-TTS-Preview-0.6b (arktts) ported to Swift-MLX.
//
// Two-module split (mirrors mlx-gepard-swift):
//   • Audio8TTSCore — the inference port: the DualAR transformer (24-layer slow AR
//                     emitting one semantic token per frame + 4-layer fast AR emitting
//                     the frame's 10 codec codebooks), the 44.1 kHz arktts codec
//                     (DAC-style encoder/decoder, split semantic+residual RVQ, windowed
//                     transformer pre/post modules), reference-exact sampling (legacy
//                     top-k/top-p order, exponential-race, RAS rescue), and the
//                     chat-format prompt builder. MLX + swift-transformers only.
//   • MLXAudio8TTS  — the engine-facing wrapper: configuration + the `tts` ModelPackage,
//                     license gate, WeightSourcing auto-materialization, CAN seams.
//
// Parity gates are CLI modes of `audio8-gates` (swift run — the SPM test product's
// metallib is unreliable for kernels). Oracle fixtures: oracle-capture/goldens/
// arktts_goldens.safetensors (45 tensors dumped from the fp32 PyTorch reference).
//
// Numerics notes carried from the Python port (see PORTING-SPEC.md):
//   – rope tables computed in Double then rounded to bf16 (the reference stores bf16
//     buffers), applied in fp32;
//   – RMSNorm applies weight AFTER the downcast to x.dtype;
//   – fast-AR codebook 0 is (semantic − semanticBeginId), never sampled;
//   – NAX split-K window never dispatches here (max FFN K = 4864 < 10240) — no
//     row-chunk workaround needed.

import PackageDescription

let package = Package(
    name: "mlx-audio8-tts-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Audio8TTSCore", targets: ["Audio8TTSCore"]),
        .library(name: "MLXAudio8TTS", targets: ["MLXAudio8TTS"]),
        .executable(name: "audio8-gates", targets: ["audio8-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.5"),
        // Qwen byte-level BPE (tokenizer.json) for the prompt builder.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.40.1"),
        // Falcon-H1 hybrid slow stack for the 0.1b checkpoint (slow_backbone: falcon_h1).
        // Same dependency mlx-gepard-swift already carries in this area.
        //
        // TEMPORARY: pinned to the fork branch carrying ml-explore/mlx-swift-lm#596, which
        // exposes the encoder surface this package needs (arktts feeds the slow stack a
        // COMPOSITE embedding, so it can never go through the token-id entry point).
        // Flip to `.package(url: "https://github.com/ml-explore/mlx-swift-lm", from: ...)`
        // once that PR merges — nothing else has to change.
        .package(
            url: "https://github.com/xocialize/mlx-swift-lm",
            branch: "falcon-h1-encoder-spi"),
    ],
    targets: [
        .target(
            name: "Audio8TTSCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/Audio8TTSCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "audio8-gates",
            dependencies: [
                "Audio8TTSCore",
                "MLXAudio8TTS",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/audio8-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXAudio8TTS",
            dependencies: [
                "Audio8TTSCore",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXAudio8TTS",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXAudio8TTSTests",
            dependencies: [
                "Audio8TTSCore",
                "MLXAudio8TTS",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformanceNN", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXAudio8TTSTests"
        ),
    ]
)
