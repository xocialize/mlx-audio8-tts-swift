// InferenceModeTests.swift — Audio8-TTS through the engine's INF gate (C14).
//
// Audio8 is the fleet's NEITHER case: no BatchNorm (no running statistics to read from the wrong
// distribution) and no Dropout (nothing to randomly zero). The LM graph is Linear + Embedding +
// a hand-rolled RMSNorm that has no `training` branch; the 44.1 kHz codec is a functional port
// (weights in a flat dictionary, not an `MLXNN.Module`) so it has no training flag at all.
//
// That makes `Audio8TTS.load`'s `train(false)` HYGIENE rather than load-bearing — and saying so
// is the point of this file. The C14 lesson from the fleet is that the flag defaults to `true`
// and the damage is invisible; a package that is currently safe should record WHY it is safe and
// assert the claim, so that adding a training-sensitive layer later fails a test instead of
// silently changing numerics.
//
// POSTURE OF RECORD: `.moduleGraph`.
//
// CHOKE POINT: `Audio8TTS.load` — the only construction funnel (the engine wrapper and the gate
// CLI both go through it).

import Foundation
import MLX
import MLXNN
import MLXServeConformance
import MLXServeConformanceNN
import XCTest

@testable import Audio8TTSCore
@testable import MLXAudio8TTS

extension Audio8Package: InferenceModeInspectable {
    public func inferenceModeFlags() -> [InferenceModeConformance.ModuleTrainingFlag] {
        InferenceModeConformance.flags(of: inferenceModeGraphs)
    }
}

final class InferenceModeTests: XCTestCase {

    /// The claim the choke-point comment makes: a freshly constructed LM graph contains no
    /// training-sensitive module type. If someone later adds a BatchNorm or a Dropout, this
    /// fails and the C14 rationale gets revisited deliberately instead of silently.
    func testGraphCarriesNoTrainingSensitiveModules() {
        let model = ArkttsModel(config: Self.tinyConfig)
        let flags = InferenceModeConformance.flags(of: ["lm": model])
        XCTAssertFalse(flags.isEmpty, "the walk observed no modules — the seam is broken")
        for flag in flags {
            XCTAssertFalse(
                flag.type.contains("BatchNorm") || flag.type.contains("Dropout"),
                "\(flag.path) is a \(flag.type): Audio8's C14 rationale says the graph carries "
                + "no training-sensitive modules. Adding one makes train(false) load-bearing — "
                + "update the choke-point comment and this test together.")
        }
    }

    /// `train(false)` at the choke point really does reach every module in the graph (the flag
    /// is recursive), so the posture holds for the whole tree rather than just the root.
    func testTrainFalseReachesEveryModule() {
        let model = ArkttsModel(config: Self.tinyConfig)
        model.train(false)
        let flags = InferenceModeConformance.flags(of: ["lm": model])
        let stillTraining = flags.filter(\.training)
        XCTAssertTrue(stillTraining.isEmpty,
                      "modules still in training mode after train(false): "
                      + stillTraining.map(\.path).joined(separator: ", "))
    }

    /// The seam reads real state: an unloaded package reports no modules, which INF-1 fails
    /// rather than passing vacuously.
    func testUnloadedPackageFailsINF1() async {
        let package = Audio8Package(configuration: Audio8Configuration())
        let report = await InferenceModeConformance.check(package, posture: .moduleGraph)
        XCTAssertFalse(report.passed,
                       "an unloaded package must not pass INF-1:\n\(report.summary)")
    }

    /// A tiny config — the gate inspects module TYPES and flags, not weights, so this needs no
    /// checkpoint and stays fast.
    static let tinyConfig = ArkttsConfig(
        vocabSize: 320, dim: 32, nLayer: 2, nHead: 4, nLocalHeads: 2, headDim: 8,
        intermediateSize: 64, maxSeqLen: 64, ropeBase: 1_000_000, normEps: 1e-6,
        attentionQkvBias: true, attentionQkNorm: false, attentionOBias: false,
        codebookSize: 16, numCodebooks: 4, semanticBeginId: 256, semanticEndId: 271,
        nFastLayer: 2, fastDim: 32, fastNHead: 4, fastNLocalHeads: 2, fastHeadDim: 8,
        fastIntermediateSize: 64, fastAttentionQkvBias: false, fastAttentionQkNorm: false,
        fastAttentionOBias: false, normFastlayerInput: true,
        codecSampleRate: 44100, codecFrameSize: 2048, codecPostNLayer: 2, codecPostNHead: 4,
        codecPostNLocalHeads: 2, codecPostIntermediateSize: 64,
        rasWindowSize: 10, rasTemperature: 1.0, rasTopP: 0.9,
        eosTokenId: 300, padTokenId: 301)
}
