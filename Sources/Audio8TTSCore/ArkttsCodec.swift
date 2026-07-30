// ArkttsCodec.swift — the 44.1 kHz arktts codec (DAC-style encoder/decoder, split
// semantic+residual RVQ, windowed transformer pre/post modules), ported 1:1 from the
// parity-locked Python-MLX reference (mlx_audio/tts/models/arktts/codec.py).
//
// Functional style (the NanoCodec pattern): weights live in a flat dictionary keyed by
// the published checkpoint names (codec. prefix stripped) — already weight-norm-folded
// and in MLX channels-last layout by the Python conversion. All tensors fp32, all
// activations (B, T, C).

import Foundation
import MLX
import MLXFast
import MLXNN

public final class ArkttsCodec: @unchecked Sendable {
    public static let sampleRate = 44100
    public static let hopLength = 512
    public static let frameLength = 2048

    let weights: [String: MLXArray]
    let config: ArkttsConfig

    /// weights: the codec.safetensors dict with the "codec." prefix stripped, fp32.
    public init(weights: [String: MLXArray], config: ArkttsConfig) {
        self.weights = weights
        self.config = config
    }

    func w(_ key: String) -> MLXArray {
        guard let value = weights[key] else { fatalError("ArkttsCodec: missing weight \(key)") }
        return value
    }

    func has(_ key: String) -> Bool { weights[key] != nil }

    // MARK: primitive ops

    static func extraPadding(length: Int, kernelSize: Int, stride: Int, paddingTotal: Int) -> Int {
        let frames = Double(length - kernelSize + paddingTotal) / Double(stride) + 1
        let ideal = (Int(frames.rounded(.up)) - 1) * stride + kernelSize - paddingTotal
        return ideal - length
    }

    /// Causal conv over (B, T, C): left-pad (k−s), right-pad to a whole frame.
    func causalConv(_ x: MLXArray, prefix: String, stride: Int = 1, dilation: Int = 1,
                    groups: Int = 1) -> MLXArray {
        let weight = w("\(prefix).conv.weight")  // (O, K, I/groups)
        let kernel = (weight.shape[1] - 1) * dilation + 1
        let padding = kernel - stride
        let right = Self.extraPadding(
            length: x.shape[1], kernelSize: kernel, stride: stride, paddingTotal: padding)
        let padded = MLX.padded(
            x, widths: [IntOrPair((0, 0)), IntOrPair((padding, right)), IntOrPair((0, 0))])
        var out = conv1d(padded, weight, stride: stride, padding: 0, dilation: dilation, groups: groups)
        if has("\(prefix).conv.bias") { out = out + w("\(prefix).conv.bias") }
        return out
    }

    /// Causal transpose conv: convTransposed then crop (k−s) from the right.
    func causalConvTranspose(_ x: MLXArray, prefix: String, stride: Int) -> MLXArray {
        let weight = w("\(prefix).conv.weight")  // (O, K, I)
        var out = convTransposed1d(x, weight, stride: stride, padding: 0)
        if has("\(prefix).conv.bias") { out = out + w("\(prefix).conv.bias") }
        let crop = weight.shape[1] - stride
        return crop > 0 ? out[0..., ..<(out.shape[1] - crop)] : out
    }

    func snake(_ x: MLXArray, prefix: String) -> MLXArray {
        let alpha = w("\(prefix).alpha")  // (1, 1, C)
        return x + (1.0 / (alpha + 1e-9)) * sin(alpha * x).square()
    }

    func rmsNorm(_ x: MLXArray, key: String, eps: Float = 1e-5) -> MLXArray {
        let xf = x.asType(.float32)
        let normalized = xf * rsqrt(xf.square().mean(axis: -1, keepDims: true) + eps)
        return normalized.asType(x.dtype) * w(key)
    }

    /// Codec rope (base 1e4), bf16-rounded like the reference's per-forward table.
    static func ropeTable(length: Int, headDim: Int, base: Double = 10000) -> MLXArray {
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
        return MLXArray(table, [length, half, 2]).asType(.bfloat16)
    }

    // MARK: window transformer

    public struct TransformerSpec {
        var nLayer: Int
        var nHead: Int
        var nLocalHeads: Int
        var dim: Int
        var headDim: Int = 64
        var windowSize: Int
    }

    public func windowTransformer(_ xIn: MLXArray, prefix: String, spec: TransformerSpec) -> MLXArray {
        var x = xIn
        let length = x.shape[1]
        let row = MLXArray(0..<Int32(length))[0..., .newAxis]
        let column = MLXArray(0..<Int32(length))[.newAxis, 0...]
        var mask = column .<= row
        mask = mask .&& (column .>= maximum(row - Int32(spec.windowSize - 1), 0))
        let fullMask = mask.expandedDimensions(axes: [0, 1])
        let rope = Self.ropeTable(length: length, headDim: spec.headDim)
        let querySize = spec.nHead * spec.headDim
        let kvSize = spec.nLocalHeads * spec.headDim

        for layer in 0..<spec.nLayer {
            let base = "\(prefix).layers.\(layer)"
            // attention
            let attnIn = rmsNorm(x, key: "\(base).attention_norm.weight")
            let qkv = matmul(attnIn, w("\(base).attention.wqkv.weight").T)
            var query = qkv[.ellipsis, 0..<querySize]
                .reshaped([x.shape[0], length, spec.nHead, spec.headDim])
            var key = qkv[.ellipsis, querySize..<(querySize + kvSize)]
                .reshaped([x.shape[0], length, spec.nLocalHeads, spec.headDim])
            let value = qkv[.ellipsis, (querySize + kvSize)...]
                .reshaped([x.shape[0], length, spec.nLocalHeads, spec.headDim])
                .transposed(0, 2, 1, 3)
            query = applyRope(query, rope).transposed(0, 2, 1, 3)
            key = applyRope(key, rope).transposed(0, 2, 1, 3)
            var attnOut = MLXFast.scaledDotProductAttention(
                queries: query, keys: key, values: value,
                scale: 1.0 / Float(Double(spec.headDim).squareRoot()), mask: fullMask)
            attnOut = attnOut.transposed(0, 2, 1, 3).reshaped([x.shape[0], length, querySize])
            attnOut = matmul(attnOut, w("\(base).attention.wo.weight").T)
            x = x + attnOut * w("\(base).attention_layer_scale.gamma")
            // feed-forward
            let ffnIn = rmsNorm(x, key: "\(base).ffn_norm.weight")
            let gate = silu(matmul(ffnIn, w("\(base).feed_forward.w1.weight").T))
            let up = matmul(ffnIn, w("\(base).feed_forward.w3.weight").T)
            let ffnOut = matmul(gate * up, w("\(base).feed_forward.w2.weight").T)
            x = x + ffnOut * w("\(base).ffn_layer_scale.gamma")
        }
        return rmsNorm(x, key: "\(prefix).norm.weight")
    }

    // MARK: encoder

    public func residualUnit(_ x: MLXArray, prefix: String, dilation: Int) -> MLXArray {
        var out = snake(x, prefix: "\(prefix).block.0")
        out = causalConv(out, prefix: "\(prefix).block.1", dilation: dilation)
        out = snake(out, prefix: "\(prefix).block.2")
        out = causalConv(out, prefix: "\(prefix).block.3")
        let difference = x.shape[1] - out.shape[1]
        let trimmed = difference > 0 ? x[0..., ..<(x.shape[1] - difference)] : x
        return trimmed + out
    }

    func encoderBlock(_ xIn: MLXArray, prefix: String, stride: Int, dim: Int,
                      transformerLayers: Int) -> MLXArray {
        var x = xIn
        for (index, dilation) in [1, 3, 9].enumerated() {
            x = residualUnit(x, prefix: "\(prefix).block.\(index)", dilation: dilation)
        }
        x = snake(x, prefix: "\(prefix).block.3")
        x = causalConv(x, prefix: "\(prefix).block.4", stride: stride)
        if transformerLayers > 0 {
            x = windowTransformer(
                x, prefix: "\(prefix).block.5",
                spec: TransformerSpec(
                    nLayer: transformerLayers, nHead: dim / 64, nLocalHeads: dim / 64,
                    dim: dim, windowSize: 512))
        }
        return x
    }

    /// audio: (B, samples, 1) → (B, frames512, 1024)
    public func encoderForward(_ audio: MLXArray) -> MLXArray {
        var x = causalConv(audio, prefix: "encoder.block.0")
        var dim = 64
        for (index, (stride, layers)) in zip([2, 4, 8, 8], [0, 0, 0, 4]).enumerated() {
            dim *= 2
            x = encoderBlock(
                x, prefix: "encoder.block.\(index + 1)", stride: stride, dim: dim,
                transformerLayers: layers)
        }
        x = snake(x, prefix: "encoder.block.5")
        return causalConv(x, prefix: "encoder.block.6")
    }

    // MARK: decoder

    func decoderBlock(_ xIn: MLXArray, prefix: String, stride: Int) -> MLXArray {
        var x = snake(xIn, prefix: "\(prefix).block.0")
        x = causalConvTranspose(x, prefix: "\(prefix).block.1", stride: stride)
        for (index, dilation) in [1, 3, 9].enumerated() {
            x = residualUnit(x, prefix: "\(prefix).block.\(index + 2)", dilation: dilation)
        }
        return x
    }

    /// latent (B, frames512, 1024) → waveform (B, samples)
    func decoderForward(_ latent: MLXArray) -> MLXArray {
        var x = causalConv(latent, prefix: "decoder.model.0")
        for (index, stride) in [8, 8, 4, 2].enumerated() {
            x = decoderBlock(x, prefix: "decoder.model.\(index + 1)", stride: stride)
        }
        x = snake(x, prefix: "decoder.model.5")
        x = causalConv(x, prefix: "decoder.model.6")
        return tanh(x)[.ellipsis, 0]
    }

    // MARK: quantizer

    /// VQ nearest-code lookup on cosine-normalized latents. latents: (B, T, D).
    func vqEncode(_ latents: MLXArray, prefix: String) -> (MLXArray, MLXArray) {
        let (batch, length) = (latents.shape[0], latents.shape[1])
        var flattened = latents.reshaped([batch * length, -1])
        flattened = flattened / maximum(
            sqrt(flattened.square().sum(axis: -1, keepDims: true)), 1e-12)
        var codebook = w("\(prefix).codebook.weight")
        codebook = codebook / maximum(sqrt(codebook.square().sum(axis: -1, keepDims: true)), 1e-12)
        let distances = flattened.square().sum(axis: 1, keepDims: true)
            - 2 * matmul(flattened, codebook.T)
            + codebook.square().sum(axis: 1, keepDims: true).T
        let indices = argMax(-distances, axis: 1).asType(.int32).reshaped([batch, length])
        return (take(w("\(prefix).codebook.weight"), indices, axis: 0), indices)
    }

    func vqOutProj(_ quantized: MLXArray, prefix: String) -> MLXArray {
        // 1x1 conv, weight (O, 1, I)
        var out = conv1d(quantized, w("\(prefix).out_proj.weight"))
        if has("\(prefix).out_proj.bias") { out = out + w("\(prefix).out_proj.bias") }
        return out
    }

    func vqForward(_ z: MLXArray, prefix: String) -> (MLXArray, MLXArray) {
        var projected = conv1d(z, w("\(prefix).in_proj.weight"))
        if has("\(prefix).in_proj.bias") { projected = projected + w("\(prefix).in_proj.bias") }
        let (quantized, indices) = vqEncode(projected, prefix: prefix)
        return (vqOutProj(quantized, prefix: prefix), indices)
    }

    func residualQuantize(_ z: MLXArray, prefix: String, count: Int) -> (MLXArray, MLXArray) {
        var quantizedSum: MLXArray? = nil
        var residual = z
        var codes = [MLXArray]()
        for index in 0..<count {
            let (quantized, indices) = vqForward(residual, prefix: "\(prefix).quantizers.\(index)")
            quantizedSum = quantizedSum.map { $0 + quantized } ?? quantized
            residual = residual - quantized
            codes.append(indices)
        }
        return (quantizedSum!, stacked(codes, axis: 1))
    }

    func residualFromCodes(_ codes: MLXArray, prefix: String) -> MLXArray {
        var output: MLXArray? = nil
        for index in 0..<codes.shape[1] {
            let embedded = take(
                w("\(prefix).quantizers.\(index).codebook.weight"), codes[0..., index], axis: 0)
            let projected = vqOutProj(embedded, prefix: "\(prefix).quantizers.\(index)")
            output = output.map { $0 + projected } ?? projected
        }
        return output!
    }

    func convNeXtBlock(_ xIn: MLXArray, prefix: String) -> MLXArray {
        var x = causalConv(xIn, prefix: "\(prefix).dwconv", groups: xIn.shape[2])
        x = MLXFast.layerNorm(x, weight: w("\(prefix).norm.weight"),
                              bias: w("\(prefix).norm.bias"), eps: 1e-6)
        x = matmul(x, w("\(prefix).pwconv1.weight").T) + w("\(prefix).pwconv1.bias")
        x = gelu(x)
        x = matmul(x, w("\(prefix).pwconv2.weight").T) + w("\(prefix).pwconv2.bias")
        return xIn + w("\(prefix).gamma") * x
    }

    func downsample(_ xIn: MLXArray) -> MLXArray {
        var x = xIn
        for stage in 0..<2 {
            x = causalConv(x, prefix: "quantizer.downsample.\(stage).0", stride: 2)
            x = convNeXtBlock(x, prefix: "quantizer.downsample.\(stage).1")
        }
        return x
    }

    func upsample(_ xIn: MLXArray) -> MLXArray {
        var x = xIn
        for stage in 0..<2 {
            x = causalConvTranspose(x, prefix: "quantizer.upsample.\(stage).0", stride: 2)
            x = convNeXtBlock(x, prefix: "quantizer.upsample.\(stage).1")
        }
        return x
    }

    public var preSpec: TransformerSpec {
        TransformerSpec(nLayer: 8, nHead: 16, nLocalHeads: 16, dim: 1024, windowSize: 128)
    }

    var postSpec: TransformerSpec {
        TransformerSpec(
            nLayer: config.codecPostNLayer, nHead: config.codecPostNHead,
            nLocalHeads: config.codecPostNLocalHeads, dim: 1024, windowSize: 128)
    }

    /// z: (B, frames512, 1024) → codes (B, 10, frames2048)
    func quantizerEncode(_ z: MLXArray) -> MLXArray {
        let down = windowTransformer(downsample(z), prefix: "quantizer.pre_module", spec: preSpec)
        let (semantic, semanticCodes) = residualQuantize(
            down, prefix: "quantizer.semantic_quantizer", count: 1)
        let (_, residualCodes) = residualQuantize(
            down - semantic, prefix: "quantizer.quantizer", count: 9)
        return concatenated([semanticCodes, residualCodes], axis: 1)
    }

    /// codes (B, 10, T) → latent (B, T*4, 1024)
    public func quantizerDecode(_ codes: MLXArray) -> MLXArray {
        upsample(postModuleLatent(codes))
    }

    // MARK: - The streaming split
    //
    // The decode path has two halves with VERY different characters, and separating them is
    // what makes a bounded decode possible here:
    //
    //   codes ─▶ RVQ + post_module ─▶ latent ─▶ upsample + decoder ─▶ waveform
    //            └─ long context,              └─ short context (11 frames),
    //               cheap (1024 × T)              expensive (expands to 2048 × T samples)
    //
    // `post_module` is 8 stacked layers of 128-wide causal attention, so its receptive field
    // COMPOUNDS to 8 × 127 = 1016 frames (~47 s). Windowing the whole stack the way
    // mlx-gepard-swift windows NanoCodec (L ≈ 26) is therefore not viable — the context would
    // exceed a typical utterance. But it is also not where the memory goes: it is a transformer
    // over 1024 × T, while the conv stack below it expands to 2048 samples PER FRAME through
    // 1536/768/384/192/96-channel intermediates. That half has a receptive field of 11 frames.
    //
    // So: run `post_module` once over the whole utterance, then window only the conv stack.
    // Same exactness argument as Gepard's, applied at the seam where it actually holds.

    /// codes → `post_module` output (B, T, 1024). The long-context, cheap half.
    public func postModuleLatent(_ codes: MLXArray) -> MLXArray {
        let semanticIndices = clip(codes[0..., ..<1], min: 0, max: 4095)
        let residualIndices = clip(codes[0..., 1...], min: 0, max: 1023)
        let semantic = residualFromCodes(semanticIndices, prefix: "quantizer.semantic_quantizer")
        let residual = residualFromCodes(residualIndices, prefix: "quantizer.quantizer")
        return windowTransformer(
            semantic + residual, prefix: "quantizer.post_module", spec: postSpec)
    }

    /// `post_module` latent (B, T, 1024) → waveform (B, T*2048). The short-context, expensive half.
    public func decodeFromLatent(_ latent: MLXArray) -> MLXArray {
        decoderForward(upsample(latent))
    }

    /// Left receptive field of `decodeFromLatent`, in LATENT FRAMES, computed from the actual
    /// loaded kernel shapes rather than hardcoded.
    ///
    /// Every op below `post_module` is strictly causal — convs left-pad only (stride-1 causal
    /// convs add no right padding), transpose convs trim right only, and the activations are
    /// pointwise. So decoding frames `[a-L, b)` and discarding the first `L` frames' samples is
    /// **bit-identical** to decoding `[0, b)` and slicing, for any `L >= leftReceptiveFieldFrames`.
    /// That exactness is the whole point: this is a windowing of the same computation, not an
    /// approximation of it.
    ///
    /// Backward extent propagation, output → input. Per decoder block (walked in reverse): the
    /// three residual units contribute `Σ (k-1)·dilation`, then the transposed up-conv converts
    /// fine → coarse as `ceil((e + k - 1) / stride)`.
    public var leftReceptiveFieldFrames: Int {
        func kernel(_ key: String) -> Int { weights[key].map { $0.shape[1] } ?? 1 }

        // Residual units: causalConv k dilation d, then k=1 → (k-1)·d each.
        let resKernel = kernel("decoder.model.1.block.2.block.1.conv.weight")
        let resExtent = [1, 3, 9].reduce(0) { $0 + (resKernel - 1) * $1 }

        var extent = kernel("decoder.model.6.conv.weight") - 1     // final causal conv
        for (index, stride) in [8, 8, 4, 2].enumerated().reversed() {
            extent += resExtent
            let k = kernel("decoder.model.\(index + 1).block.1.conv.weight")
            extent = Int((Double(extent + k - 1) / Double(stride)).rounded(.up))
        }
        extent += kernel("decoder.model.0.conv.weight") - 1        // decoder input conv
        // Now in latent positions (4×frames). Walk back through the two upsample stages.
        for stage in [1, 0] {
            extent += kernel("quantizer.upsample.\(stage).1.dwconv.conv.weight") - 1
            let k = kernel("quantizer.upsample.\(stage).0.conv.weight")
            extent = Int((Double(extent + k - 1) / Double(2)).rounded(.up))
        }
        return extent
    }

    /// The smallest chunk size at which windowed decode is **bit-identical** to the
    /// whole-utterance decode, measured (`--s5`), not derived.
    ///
    /// The windowing is algorithmically exact at `leftReceptiveFieldFrames` of context — the
    /// context sweep proves that, since raising context from 11 to 128 frames changes the
    /// result not at all. What does change it is the SIZE of the decoded window: MLX selects
    /// different conv kernels for small inputs, and their fp32 reduction order differs from the
    /// path a full-length decode takes. Measured on a 102-frame utterance:
    ///
    ///     chunk  8 → 1.4e-3      chunk 48 → 0.0
    ///     chunk 16 → 1.4e-3      chunk 64 → 0.0
    ///     chunk 32 → 2.5e-4      chunk 96 → 0.0
    ///
    /// So the rule is a floor on chunk size, not on context. 1.4e-3 on a [-1,1] waveform is
    /// about −57 dBFS and inaudible, but "bit-identical" is a claim worth keeping true rather
    /// than approximately true — hence the default sits above the threshold.
    public var minimumExactChunkFrames: Int { 48 }

    /// Windowed, incremental decode. Emits each chunk's samples through `onChunk` as soon as it
    /// is decoded, and never holds more than `context + chunkFrames` frames in the conv stack —
    /// so peak activation is bounded by the chunk size instead of growing with utterance length.
    ///
    /// - Parameters:
    ///   - codes: (B, 10, T). Batch 1.
    ///   - chunkFrames: frames emitted per chunk. **Keep this ≥ `minimumExactChunkFrames`**
    ///     (48) — see that property for why smaller chunks stop being bit-exact.
    ///   - contextFrames: left context. Defaults to `leftReceptiveFieldFrames`, the smallest
    ///     value that is structurally sufficient. Lower it only to demonstrate the drift.
    ///   - onChunk: `(samples, isFinal)`, called in order.
    public func decodeStreaming(_ codes: MLXArray,
                                chunkFrames: Int = 64,
                                contextFrames: Int? = nil,
                                onChunk: (MLXArray, Bool) throws -> Void) rethrows {
        let latent = postModuleLatent(codes.asType(.int32))
        eval(latent)
        let total = latent.shape[1]
        guard total > 0 else { return }
        let context = max(0, contextFrames ?? leftReceptiveFieldFrames)
        let step = max(1, chunkFrames)

        // Never emit a degenerate final chunk: if the remainder after this one would be
        // smaller than `minTail`, absorb it here instead. Two reasons — a tiny decode is pure
        // overhead (it still pays the full `context` re-decode), and MLX selects different conv
        // kernels for very small inputs, so a stub window is where windowed output stops being
        // bit-reproducible against the whole-utterance decode on the GPU stream.
        let minTail = max(8, step / 4)
        var start = 0
        while start < total {
            var end = min(start + step, total)
            if total - end < minTail { end = total }
            let windowStart = max(0, start - context)
            let window = latent[0..., windowStart ..< end]
            let decoded = decodeFromLatent(window)
            // Drop the contaminated prefix: those positions were computed against zero-padding
            // where the full decode had real context.
            let drop = (start - windowStart) * Self.frameLength
            let samples = decoded[0..., drop...]
            eval(samples)
            try onChunk(samples, end >= total)
            start = end
        }
    }

    // MARK: public surface

    /// audio: (B, samples) mono in [−1, 1] → (codes (B, 10, frames) int32, frames)
    public func encode(_ audioIn: MLXArray, sampleCount: Int) -> (MLXArray, Int) {
        var audio = audioIn[.ellipsis, .newAxis]
        let original = audio.shape[1]
        let right = (original + Self.frameLength - 1) / Self.frameLength * Self.frameLength - original
        if right > 0 {
            audio = padded(
                audio, widths: [IntOrPair((0, 0)), IntOrPair((0, right)), IntOrPair((0, 0))])
        }
        let encoded = encoderForward(audio)
        let codes = quantizerEncode(encoded)
        let codeLength = min(
            (sampleCount + Self.frameLength - 1) / Self.frameLength, codes.shape[2])
        return (codes[0..., 0..., ..<codeLength], codeLength)
    }

    /// codes: (B, 10, T) → waveform (B, T*2048)
    public func decode(_ codes: MLXArray) -> MLXArray {
        decoderForward(quantizerDecode(codes.asType(.int32)))
    }
}
