// Generator.swift — reference-exact sampling (semantic filter → legacy top-k/top-p →
// temperature → exponential race, with RAS repetition rescue) and the DualAR
// generation loop, ported 1:1 from the parity-locked Python-MLX reference.

import Foundation
import MLX
import MLXRandom

public struct SamplingParams: Sendable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var doSample: Bool
    public var seed: UInt64?

    public init(maxNewTokens: Int = 512, temperature: Float = 0.7, topP: Float = 0.9,
                topK: Int = 50, doSample: Bool = true, seed: UInt64? = nil) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.doSample = doSample
        self.seed = seed
    }
}

extension ArkttsModel {

    // MARK: sampling primitives

    /// Legacy filter order: remove where cumulative softmax > topP OR rank >= topK; keep rank 0.
    static func legacyTopKTopP(_ scores: MLXArray, topK: Int, topP: Float) -> MLXArray {
        let sortedIndices = argSort(-scores, axis: -1)
        let sortedScores = takeAlong(scores, sortedIndices, axis: -1)
        let cumulative = cumsum(softmax(sortedScores, axis: -1), axis: -1)
        let positions = MLXArray(0..<Int32(scores.shape[scores.ndim - 1]))
        var removeSorted = (cumulative .> topP) .|| (positions[.newAxis, 0...] .>= Int32(topK))
        removeSorted[0..., 0] = MLXArray(false)
        var remove = MLXArray.zeros(removeSorted.shape, dtype: .bool)
        remove = putAlong(remove, sortedIndices, values: removeSorted, axis: -1)
        return which(remove, MLXArray(-Float.infinity).asType(scores.dtype), scores)
    }

    func semanticFilter(_ scores: MLXArray) -> MLXArray {
        var filtered = MLXArray.full(scores.shape, values: MLXArray(-Float.infinity))
            .asType(scores.dtype)
        filtered[0..., config.semanticBeginId...config.semanticEndId] =
            scores[0..., config.semanticBeginId...config.semanticEndId]
        filtered[0..., config.eosTokenId] = scores[0..., config.eosTokenId]
        return filtered
    }

    /// Exponential race: argmax(p / -log U) samples from the categorical p.
    static func raceSample(_ scores: MLXArray, key: MLXArray) -> MLXArray {
        let probabilities = softmax(scores, axis: -1)
        let random = MLXRandom.uniform(low: Float(0), high: Float(1), probabilities.shape, key: key)
        let noise = -log(random)
        return argMax(probabilities / noise, axis: -1).asType(.int32)
    }

    func processedScores(
        _ scoresIn: MLXArray, topK: Int, topP: Float, temperature: Float, semantic: Bool
    ) -> MLXArray {
        var scores = scoresIn
        if semantic { scores = semanticFilter(scores) }
        scores = Self.legacyTopKTopP(scores, topK: topK, topP: topP)
        return scores / max(temperature, 1e-5)
    }

    func sampleSemantic(
        _ logits: MLXArray, params: SamplingParams, previous: MLXArray?,
        keys: (MLXArray, MLXArray)
    ) -> MLXArray {
        let regular = processedScores(
            logits, topK: params.topK, topP: params.topP,
            temperature: params.temperature, semantic: true)
        if !params.doSample {
            return argMax(regular, axis: -1).asType(.int32)
        }
        let normal = Self.raceSample(regular, key: keys.0)
        let high = Self.raceSample(
            processedScores(
                logits, topK: params.topK, topP: config.rasTopP,
                temperature: config.rasTemperature, semantic: true),
            key: keys.1)
        guard let previous else { return normal }
        let repeated = (previous .== normal[0..., .newAxis]).any(axis: 1)
        let semantic = (normal .>= Int32(config.semanticBeginId))
            .&& (normal .<= Int32(config.semanticEndId))
        return which(repeated .&& semantic, high, normal)
    }

    func generateCodebooks(
        slowHidden: MLXArray, semantic: MLXArray, params: SamplingParams, key keyIn: MLXArray
    ) -> MLXArray {
        // fast_project_in is Identity for this config
        _ = fastStep(slowHidden, position: 0)
        var current = clip(
            semantic - Int32(config.semanticBeginId), min: 0, max: Int32(config.codebookSize - 1))
        var codebooks = [current]
        var hidden = fastEmbeddings(current).expandedDimensions(axis: 1)
        var key = keyIn
        for position in 1..<config.numCodebooks {
            var scores = fastStep(hidden, position: position)
            scores = processedScores(
                scores, topK: params.topK, topP: params.topP,
                temperature: params.temperature, semantic: false)
            if params.doSample {
                let split = MLXRandom.split(key: key)
                key = split.0
                current = Self.raceSample(scores, key: split.1)
            } else {
                current = argMax(scores, axis: -1).asType(.int32)
            }
            codebooks.append(current)
            hidden = fastEmbeddings(current).expandedDimensions(axis: 1)
        }
        return stacked(codebooks, axis: 1)
    }

    // MARK: prompt assembly (mirrors _prepare_prompt; batch size 1)

    /// Builds the (1, numCodebooks+1, T) prompt and its mask from token rows and
    /// optional reference codes (numCodebooks, refLength).
    public func preparePrompt(
        prefix: [Int32], suffix: [Int32], referenceCodes: MLXArray?, referenceLength: Int
    ) -> (MLXArray, MLXArray) {
        let rows: MLXArray
        if let codes = referenceCodes {
            let clipped = codes[0..., ..<referenceLength].asType(.int32)
            let semanticCodes = clipped[0] + Int32(config.semanticBeginId)
            let semanticRow = concatenated(
                [MLXArray(prefix), semanticCodes, MLXArray(suffix)], axis: 0)
            let total = semanticRow.shape[0]
            var values = MLXArray.zeros([config.numCodebooks + 1, total], dtype: .int32)
            values[0] = semanticRow
            values[1..., prefix.count..<(prefix.count + referenceLength)] = clipped
            rows = values
        } else {
            let semanticRow = concatenated([MLXArray(prefix), MLXArray(suffix)], axis: 0)
            var values = MLXArray.zeros([config.numCodebooks + 1, semanticRow.shape[0]], dtype: .int32)
            values[0] = semanticRow
            rows = values
        }
        // single row: no left-padding needed, mask all ones (row 0 pad fill is only
        // reachable with batch > 1 in the reference)
        let prompt = rows.expandedDimensions(axis: 0)
        let mask = MLXArray.ones([1, rows.shape[1]], dtype: .int32)
        return (prompt, mask)
    }

    // MARK: generation loop

    /// Returns generated codec codes (1, numCodebooks, frames), −1-free (truncated at EOS).
    public func generateCodes(
        prefix: [Int32], suffix: [Int32],
        referenceCodes: MLXArray? = nil, referenceLength: Int = 0,
        params: SamplingParams = SamplingParams(),
        onFrame: ((Int) -> Bool)? = nil
    ) -> MLXArray {
        let (prompt, promptMask) = preparePrompt(
            prefix: prefix, suffix: suffix,
            referenceCodes: referenceCodes, referenceLength: referenceLength)
        let promptWidth = prompt.shape[2]
        precondition(promptWidth < config.maxSeqLen, "prompt length \(promptWidth) >= max_seq_len")
        let maxNewTokens = min(params.maxNewTokens, config.maxSeqLen - promptWidth)
        let dtype = embeddings.weight.dtype
        setupGenerationCaches(batch: 1, dtype: dtype)
        defer { clearGenerationCaches() }

        var key = MLXRandom.key(params.seed ?? UInt64.random(in: 0..<UInt64.max))
        let positionIds = maximum(cumsum(promptMask, axis: -1) - 1, 0)
        var (logits, slowHidden) = slowStep(
            prompt, cacheStart: 0, cachePositions: MLXArray(0..<Int32(promptWidth)),
            positionIds: positionIds, attentionMask: promptMask)
        var mask = promptMask
        var previous: MLXArray? = nil
        var frames = [MLXArray]()

        for step in 0..<maxNewTokens {
            let keys = MLXRandom.split(key: key, into: 4)
            key = keys[3]
            let semantic = sampleSemantic(
                logits, params: params, previous: previous, keys: (keys[0], keys[1]))
            let codebooks = generateCodebooks(
                slowHidden: slowHidden, semantic: semantic, params: params, key: keys[2])
            eval(semantic, codebooks)
            if semantic.item(Int32.self) == Int32(config.eosTokenId) {
                break
            }
            frames.append(codebooks)
            if previous == nil {
                previous = MLXArray.zeros([1, config.rasWindowSize], dtype: .int32)
            } else {
                previous = concatenated([previous![0..., 1...], semantic[0..., .newAxis]], axis: 1)
            }
            if let onFrame, !onFrame(frames.count) {
                break
            }

            let nextColumn = concatenated([semantic[0..., .newAxis], codebooks], axis: 1)[.ellipsis, .newAxis]
            mask = concatenated([mask, MLXArray.ones([1, 1], dtype: .int32)], axis: 1)
            let physical = promptWidth + step
            let tokenPosition = MLXArray([Int32(promptWidth + step)]).expandedDimensions(axis: 0)
            (logits, slowHidden) = slowStep(
                nextColumn, cacheStart: physical, cachePositions: MLXArray([Int32(physical)]),
                positionIds: tokenPosition, attentionMask: mask)
        }

        if frames.isEmpty {
            return MLXArray.zeros([1, config.numCodebooks, 0], dtype: .int32)
        }
        return stacked(frames, axis: 2)
    }
}
