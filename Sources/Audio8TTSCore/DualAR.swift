// DualAR.swift — the arktts language model: 24-layer slow AR (one semantic token per
// audio frame) + 4-layer fast AR (the frame's 10 codec codebooks), ported 1:1 from the
// parity-locked Python-MLX reference (mlx_audio/tts/models/arktts/arktts.py).
//
// Key paths match the published checkpoint's `model.`-stripped keys exactly:
//   layers.N.attention.{wqkv,wo}.{weight,bias}, layers.N.{attention_norm,ffn_norm}.weight,
//   layers.N.feed_forward.{w1,w2,w3}.weight, embeddings.weight, codebook_embeddings.weight,
//   norm.weight, fast_layers.N…, fast_embeddings.weight, fast_norm.weight, fast_output.weight.
//
// Numerics contract (PORTING-SPEC.md): rope tables Double→bf16, applied fp32; RMSNorm
// weight multiplies AFTER downcast; KV cache is a full max_seq_len buffer with bool masks.

import Foundation
import MLX
import MLXFast
import MLXNN

public struct ArkttsConfig: Codable, Sendable {
    public var vocabSize: Int
    public var dim: Int
    public var nLayer: Int
    public var nHead: Int
    public var nLocalHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var maxSeqLen: Int
    public var ropeBase: Double
    public var normEps: Double
    public var attentionQkvBias: Bool
    public var attentionQkNorm: Bool
    public var attentionOBias: Bool
    public var codebookSize: Int
    public var numCodebooks: Int
    public var semanticBeginId: Int
    public var semanticEndId: Int
    public var nFastLayer: Int
    public var fastDim: Int
    public var fastNHead: Int
    public var fastNLocalHeads: Int
    public var fastHeadDim: Int
    public var fastIntermediateSize: Int
    public var fastAttentionQkvBias: Bool
    public var fastAttentionQkNorm: Bool
    public var fastAttentionOBias: Bool
    public var normFastlayerInput: Bool
    public var codecSampleRate: Int
    public var codecFrameSize: Int
    public var codecPostNLayer: Int
    public var codecPostNHead: Int
    public var codecPostNLocalHeads: Int
    public var codecPostIntermediateSize: Int
    public var rasWindowSize: Int
    public var rasTemperature: Float
    public var rasTopP: Float
    public var eosTokenId: Int
    public var padTokenId: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case dim
        case nLayer = "n_layer"
        case nHead = "n_head"
        case nLocalHeads = "n_local_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxSeqLen = "max_seq_len"
        case ropeBase = "rope_base"
        case normEps = "norm_eps"
        case attentionQkvBias = "attention_qkv_bias"
        case attentionQkNorm = "attention_qk_norm"
        case attentionOBias = "attention_o_bias"
        case codebookSize = "codebook_size"
        case numCodebooks = "num_codebooks"
        case semanticBeginId = "semantic_begin_id"
        case semanticEndId = "semantic_end_id"
        case nFastLayer = "n_fast_layer"
        case fastDim = "fast_dim"
        case fastNHead = "fast_n_head"
        case fastNLocalHeads = "fast_n_local_heads"
        case fastHeadDim = "fast_head_dim"
        case fastIntermediateSize = "fast_intermediate_size"
        case fastAttentionQkvBias = "fast_attention_qkv_bias"
        case fastAttentionQkNorm = "fast_attention_qk_norm"
        case fastAttentionOBias = "fast_attention_o_bias"
        case normFastlayerInput = "norm_fastlayer_input"
        case codecSampleRate = "codec_sample_rate"
        case codecFrameSize = "codec_frame_size"
        case codecPostNLayer = "codec_post_n_layer"
        case codecPostNHead = "codec_post_n_head"
        case codecPostNLocalHeads = "codec_post_n_local_heads"
        case codecPostIntermediateSize = "codec_post_intermediate_size"
        case rasWindowSize = "ras_window_size"
        case rasTemperature = "ras_temperature"
        case rasTopP = "ras_top_p"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public static func load(from url: URL) throws -> ArkttsConfig {
        try JSONDecoder().decode(ArkttsConfig.self, from: Data(contentsOf: url))
    }
}

// MARK: - RoPE

/// Precomputed (real, imag) rotation table, bf16-rounded like the reference buffer.
/// Boxed outside Module reflection so it never pollutes the parameter key set.
public final class RopeTable: @unchecked Sendable {
    public let values: MLXArray  // (length, headDim/2, 2) bf16

    public init(length: Int, headDim: Int, base: Double) {
        let half = headDim / 2
        var table = [Float]()
        table.reserveCapacity(length * half * 2)
        for position in 0..<length {
            for index in 0..<half {
                let frequency = 1.0 / pow(base, Double(2 * index) / Double(headDim))
                let phase = Double(position) * frequency
                table.append(Float(cos(phase)))
                table.append(Float(sin(phase)))
            }
        }
        self.values = MLXArray(table, [length, half, 2]).asType(.bfloat16)
    }
}

/// Pairwise rotation in fp32, cast back to x dtype. x: (B, T, H, D); rope: (T, D/2, 2) or (B, T, D/2, 2).
func applyRope(_ x: MLXArray, _ rope: MLXArray) -> MLXArray {
    let shaped = x.asType(.float32).reshaped(Array(x.shape.dropLast()) + [-1, 2])
    var r = rope.asType(.float32)
    if r.ndim == 3 {
        r = r.expandedDimensions(axes: [0, 2])
    } else {
        r = r.expandedDimensions(axis: 2)
    }
    let real = shaped[.ellipsis, 0] * r[.ellipsis, 0] - shaped[.ellipsis, 1] * r[.ellipsis, 1]
    let imag = shaped[.ellipsis, 1] * r[.ellipsis, 0] + shaped[.ellipsis, 0] * r[.ellipsis, 1]
    return stacked([real, imag], axis: -1).flattened(start: 3).asType(x.dtype)
}

// MARK: - Norm

public final class ArkttsRMSNorm: Module {
    public let eps: Float
    public var weight: MLXArray

    public init(dim: Int, eps: Double) {
        self.eps = Float(eps)
        self.weight = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let normalized = xf * rsqrt(xf.square().mean(axis: -1, keepDims: true) + eps)
        // reference applies weight AFTER the downcast to x.dtype
        return normalized.asType(x.dtype) * weight
    }
}

// MARK: - KV cache (full-buffer, isomorphic to the reference)

public final class ArkttsKVCache {
    var keys: MLXArray
    var values: MLXArray

    init(batch: Int, maxLength: Int, heads: Int, headDim: Int, dtype: DType) {
        keys = MLXArray.zeros([batch, heads, maxLength, headDim], dtype: dtype)
        values = MLXArray.zeros([batch, heads, maxLength, headDim], dtype: dtype)
    }

    func update(start: Int, k: MLXArray, v: MLXArray) -> (MLXArray, MLXArray) {
        let length = k.shape[2]
        keys[0..., 0..., start..<(start + length), 0...] = k
        values[0..., 0..., start..<(start + length), 0...] = v
        return (keys, values)
    }
}

// MARK: - Attention / FFN / Block

public final class ArkttsAttention: Module {
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    let nHead: Int
    let nLocalHeads: Int
    let headDim: Int
    var kvCache: ArkttsKVCache?

    public init(dim: Int, nHead: Int, nLocalHeads: Int, headDim: Int,
                qkvBias: Bool, outputBias: Bool) {
        let total = (nHead + 2 * nLocalHeads) * headDim
        self._wqkv.wrappedValue = Linear(dim, total, bias: qkvBias)
        self._wo.wrappedValue = Linear(nHead * headDim, dim, bias: outputBias)
        self.nHead = nHead
        self.nLocalHeads = nLocalHeads
        self.headDim = headDim
    }

    public func callAsFunction(
        _ x: MLXArray, rope: MLXArray, mask: MLXArray, cacheStart: Int? = nil
    ) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let querySize = nHead * headDim
        let kvSize = nLocalHeads * headDim
        let qkv = wqkv(x)
        var query = qkv[.ellipsis, 0..<querySize]
        var key = qkv[.ellipsis, querySize..<(querySize + kvSize)]
        var value = qkv[.ellipsis, (querySize + kvSize)...]
        query = query.reshaped([batch, length, nHead, headDim])
        key = key.reshaped([batch, length, nLocalHeads, headDim])
        value = value.reshaped([batch, length, nLocalHeads, headDim])
        query = applyRope(query, rope).transposed(0, 2, 1, 3)
        key = applyRope(key, rope).transposed(0, 2, 1, 3)
        value = value.transposed(0, 2, 1, 3)
        if let cache = kvCache {
            guard let start = cacheStart else {
                fatalError("cacheStart is required when KV cache is enabled")
            }
            (key, value) = cache.update(start: start, k: key, v: value)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value,
            scale: 1.0 / Float(Double(headDim).squareRoot()), mask: mask
        )
        return wo(output.transposed(0, 2, 1, 3).reshaped([batch, length, querySize]))
    }
}

public final class ArkttsFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(dim: Int, intermediateSize: Int) {
        self._w1.wrappedValue = Linear(dim, intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(intermediateSize, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, intermediateSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

public final class ArkttsTransformerBlock: Module {
    @ModuleInfo(key: "attention") public var attention: ArkttsAttention
    @ModuleInfo(key: "feed_forward") public var feedForward: ArkttsFeedForward
    @ModuleInfo(key: "ffn_norm") public var ffnNorm: ArkttsRMSNorm
    @ModuleInfo(key: "attention_norm") public var attentionNorm: ArkttsRMSNorm

    public init(dim: Int, intermediateSize: Int, nHead: Int, nLocalHeads: Int, headDim: Int,
                qkvBias: Bool, outputBias: Bool, normEps: Double) {
        self._attention.wrappedValue = ArkttsAttention(
            dim: dim, nHead: nHead, nLocalHeads: nLocalHeads, headDim: headDim,
            qkvBias: qkvBias, outputBias: outputBias)
        self._feedForward.wrappedValue = ArkttsFeedForward(dim: dim, intermediateSize: intermediateSize)
        self._ffnNorm.wrappedValue = ArkttsRMSNorm(dim: dim, eps: normEps)
        self._attentionNorm.wrappedValue = ArkttsRMSNorm(dim: dim, eps: normEps)
    }

    public func callAsFunction(
        _ x: MLXArray, rope: MLXArray, mask: MLXArray, cacheStart: Int? = nil
    ) -> MLXArray {
        let hidden = x + attention(attentionNorm(x), rope: rope, mask: mask, cacheStart: cacheStart)
        return hidden + feedForward(ffnNorm(hidden))
    }
}

// MARK: - The DualAR model

public final class ArkttsModel: Module {
    public let config: ArkttsConfig
    @ModuleInfo(key: "embeddings") public var embeddings: Embedding
    @ModuleInfo(key: "codebook_embeddings") public var codebookEmbeddings: Embedding
    @ModuleInfo(key: "layers") public var layers: [ArkttsTransformerBlock]
    @ModuleInfo(key: "norm") public var norm: ArkttsRMSNorm
    @ModuleInfo(key: "fast_embeddings") public var fastEmbeddings: Embedding
    @ModuleInfo(key: "fast_layers") public var fastLayers: [ArkttsTransformerBlock]
    @ModuleInfo(key: "fast_norm") public var fastNorm: ArkttsRMSNorm
    @ModuleInfo(key: "fast_output") public var fastOutput: Linear

    public let freqsCis: RopeTable
    public let fastFreqsCis: RopeTable

    public init(config: ArkttsConfig) {
        self.config = config
        precondition(config.fastDim == config.dim, "fast_project_in != Identity is not supported")
        self._embeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.dim)
        self._codebookEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.codebookSize * config.numCodebooks, dimensions: config.dim)
        self._layers.wrappedValue = (0..<config.nLayer).map { _ in
            ArkttsTransformerBlock(
                dim: config.dim, intermediateSize: config.intermediateSize,
                nHead: config.nHead, nLocalHeads: config.nLocalHeads, headDim: config.headDim,
                qkvBias: config.attentionQkvBias, outputBias: config.attentionOBias,
                normEps: config.normEps)
        }
        self._norm.wrappedValue = ArkttsRMSNorm(dim: config.dim, eps: config.normEps)
        self._fastEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.codebookSize, dimensions: config.fastDim)
        self._fastLayers.wrappedValue = (0..<config.nFastLayer).map { _ in
            ArkttsTransformerBlock(
                dim: config.fastDim, intermediateSize: config.fastIntermediateSize,
                nHead: config.fastNHead, nLocalHeads: config.fastNLocalHeads,
                headDim: config.fastHeadDim,
                qkvBias: config.fastAttentionQkvBias, outputBias: config.fastAttentionOBias,
                normEps: config.normEps)
        }
        self._fastNorm.wrappedValue = ArkttsRMSNorm(dim: config.fastDim, eps: config.normEps)
        self._fastOutput.wrappedValue = Linear(config.fastDim, config.codebookSize, bias: false)
        self.freqsCis = RopeTable(length: config.maxSeqLen, headDim: config.headDim, base: config.ropeBase)
        self.fastFreqsCis = RopeTable(
            length: config.numCodebooks, headDim: config.fastHeadDim, base: config.ropeBase)
    }

    // MARK: embedding

    /// input_ids: (B, numCodebooks+1, T) int32
    public func embed(_ inputIds: MLXArray) -> MLXArray {
        var codebookEmbeds = [MLXArray]()
        for index in 0..<config.numCodebooks {
            codebookEmbeds.append(
                codebookEmbeddings(inputIds[0..., index + 1] + Int32(index * config.codebookSize)))
        }
        var codebookSum = stacked(codebookEmbeds, axis: 1).sum(axis: 1)
        let semanticRow = inputIds[0..., 0]
        let semantic = (semanticRow .>= Int32(config.semanticBeginId))
            .&& (semanticRow .<= Int32(config.semanticEndId))
        codebookSum = which(semantic[.ellipsis, .newAxis], codebookSum, MLXArray(Float(0)))
        return embeddings(semanticRow) + codebookSum
    }

    static func causalMask(
        attentionMask: MLXArray, queryPositions: MLXArray, keyLength: Int
    ) -> MLXArray {
        var mask = attentionMask
        if mask.shape[1] < keyLength {
            mask = padded(mask, widths: [IntOrPair((0, 0)), IntOrPair((0, keyLength - mask.shape[1]))])
        }
        let keyPositions = MLXArray(0..<Int32(keyLength))
        let causal = keyPositions[.newAxis, 0...] .<= queryPositions[0..., .newAxis]
        return causal.expandedDimensions(axes: [0, 1])
            .&& mask[0..., ..<keyLength].expandedDimensions(axes: [1, 2]).asType(.bool)
    }

    // MARK: prefill (no-cache) forward — parity surface

    /// Returns (logits, hidden) like the reference forward.
    public func callAsFunction(
        _ inputIds: MLXArray, attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (batch, length) = (inputIds.shape[0], inputIds.shape[2])
        let mask2d = attentionMask ?? MLXArray.ones([batch, length], dtype: .int32)
        let positionIds = maximum(cumsum(mask2d.asType(.int32), axis: -1) - 1, 0)
        let rope = freqsCis.values[positionIds]
        let mask = Self.causalMask(
            attentionMask: mask2d, queryPositions: MLXArray(0..<Int32(length)), keyLength: length)
        var hidden = embed(inputIds)
        for layer in layers {
            hidden = layer(hidden, rope: rope, mask: mask)
        }
        let normalized = norm(hidden)
        let logits = matmul(normalized, embeddings.weight.T)
        return (logits, hidden)
    }

    // MARK: cached decode

    public func setupGenerationCaches(batch: Int, dtype: DType) {
        for layer in layers {
            layer.attention.kvCache = ArkttsKVCache(
                batch: batch, maxLength: config.maxSeqLen,
                heads: config.nLocalHeads, headDim: config.headDim, dtype: dtype)
        }
        for layer in fastLayers {
            layer.attention.kvCache = ArkttsKVCache(
                batch: batch, maxLength: config.numCodebooks,
                heads: config.fastNLocalHeads, headDim: config.fastHeadDim, dtype: dtype)
        }
    }

    public func clearGenerationCaches() {
        for layer in layers { layer.attention.kvCache = nil }
        for layer in fastLayers { layer.attention.kvCache = nil }
    }

    /// One slow step over the (possibly multi-token) column. Returns (logits, fastHidden).
    public func slowStep(
        _ inputIds: MLXArray, cacheStart: Int, cachePositions: MLXArray,
        positionIds: MLXArray, attentionMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        var hidden = embed(inputIds)
        let rope = freqsCis.values[positionIds]
        let mask = Self.causalMask(
            attentionMask: attentionMask, queryPositions: cachePositions, keyLength: config.maxSeqLen)
        for layer in layers {
            hidden = layer(hidden, rope: rope, mask: mask, cacheStart: cacheStart)
        }
        hidden = hidden[0..., (hidden.shape[1] - 1)...]
        let normalized = norm(hidden)
        let logits = matmul(normalized, embeddings.weight.T)[0..., -1]
        let fastHidden = config.normFastlayerInput ? normalized : hidden
        return (logits, fastHidden)
    }

    public func fastStep(_ hiddenIn: MLXArray, position: Int) -> MLXArray {
        var hidden = hiddenIn
        let rope = fastFreqsCis.values[MLXArray([Int32(position)])]
        let keyMask = MLXArray.ones([hidden.shape[0], config.numCodebooks], dtype: .bool)
        let mask = Self.causalMask(
            attentionMask: keyMask, queryPositions: MLXArray([Int32(position)]),
            keyLength: config.numCodebooks)
        for layer in fastLayers {
            hidden = layer(hidden, rope: rope, mask: mask, cacheStart: position)
        }
        return fastOutput(fastNorm(hidden))[0..., -1]
    }
}
