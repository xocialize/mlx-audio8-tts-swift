// Conformance for `Audio8MiniPackage` (Audio8-TTS-Preview-0.1b).
//
// Mirrors ConformanceTests.swift, with the assertions that must genuinely differ written to
// assert what is TRUE rather than copied across. Two of them are the point of this file:
//
//   • the licence is NOT permissive — copying testLicenseIsPermissiveOnBothLayers would have
//     been a passing test asserting a falsehood about a revenue-capped licence;
//   • the resident floor is ~800 MB LOWER than the 0.6b's, so the 0.6b's "below the weight
//     bytes — was it measured?" threshold would pass this package vacuously.

import Foundation
import MLXToolKit
import MLXServeConformance
import MLXServeCore
import XCTest

@testable import MLXAudio8TTS

final class MiniManifestConformanceTests: XCTestCase {

    /// C7/C8. The weight layer is deliberately NOT on `permissiveAllowlist`: the Audio8
    /// Community Licence permits commercial use only below US$2M annual revenue. The engine
    /// therefore CLASSIFIES it (a `LicenseAdvisory`) rather than admitting it silently, which
    /// since contract 1.28.0 reports instead of blocking.
    ///
    /// This test is written to fail loudly if that ever flips by accident — but to keep passing
    /// once the engine-side allowlisting on `mlx-engine-swift@audio8-community-license` merges,
    /// since that is a deliberate change with its own review. What it pins unconditionally is
    /// that BOTH layers are declared (which is what C7/C8 actually assert) and that the weight
    /// layer is not silently claimed to be Apache-2.0 like its 0.6b sibling.
    func testBothLicenseLayersAreDeclaredAndWeightLayerIsTheCappedOne() {
        let license = Audio8MiniPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .audio8Community,
                       "the LM weights carry the revenue-capped community licence, not the "
                       + "0.6b's Apache-2.0")
        XCTAssertEqual(license.portCodeLicense, .apache2)
        XCTAssertNotEqual(license.weightLicense, Audio8Package.manifest.license.weightLicense,
                          "the two checkpoints' weight licences genuinely differ — if these ever "
                          + "match, one of the manifests is wrong")
    }

    /// The footprint must be MEASURED, and measured for THIS checkpoint. The bound below is
    /// derived from what the weights actually are: ~340 MB bf16 LM + ~1.35 GB fp32 codec.
    /// A declaration under that could not have come from a real load.
    func testFootprintIsSplitAndMeasuredForThisCheckpoint() {
        guard let footprint = Audio8MiniPackage.manifest.requirements.footprints
            .first(where: { $0.quant == .bf16 }) else {
            return XCTFail("no bf16 footprint declared")
        }
        XCTAssertGreaterThan(footprint.residentBytes, 1_600_000_000,
                             "resident floor is below the weight bytes (~340 MB LM + ~1.35 GB "
                             + "codec) — was it measured?")
        XCTAssertGreaterThan(footprint.peakActivationBytes, 0,
                             "activation peak must be declared separately from the resident floor")
        // Measured transient was 3711 MB; the declaration must cover it.
        XCTAssertGreaterThan(footprint.peakActivationBytes, 3_711_000_000,
                             "declared activation is below the measured 3711 MB transient")
    }

    /// The whole reason this package exists: it is lighter. If it ever stops being lighter than
    /// its sibling, the tradeoff it asks callers to accept (worse speaker similarity, fewer
    /// languages) buys nothing.
    func testItIsActuallyLighterThanTheSixHundredMillionSibling() {
        let mini = Audio8MiniPackage.manifest.requirements.footprints.first { $0.quant == .bf16 }
        let full = Audio8Package.manifest.requirements.footprints.first { $0.quant == .bf16 }
        guard let mini, let full else { return XCTFail("both packages must declare a bf16 footprint") }
        XCTAssertLessThan(mini.residentBytes, full.residentBytes,
                          "the compact checkpoint must have a lower resident floor")
    }

    /// Strength is a selection signal. This package clones measurably less faithfully
    /// (Seed-TTS SIM 56.7/68.2 vs 63.2/73.1), so it must not claim the specialty as strongly —
    /// otherwise a selector reads the two as interchangeable for the one job they are picked for.
    func testVoiceCloneClaimIsWeakerThanTheSibling() {
        func strength(_ specialties: [SpecialtyWeight]) -> Double? {
            specialties.first { $0.specialty == .voiceClone }?.strength
        }
        guard let mini = strength(Audio8MiniPackage.manifest.specialties),
              let full = strength(Audio8Package.manifest.specialties) else {
            return XCTFail("both packages must declare a voiceClone specialty")
        }
        XCTAssertLessThan(mini, full,
                          "the lower-similarity checkpoint must not claim voiceClone as strongly")
    }

    func testSpecialtiesAreRegistered() {
        for weight in Audio8MiniPackage.manifest.specialties {
            XCTAssertTrue(weight.specialty.isRegistered,
                          "\(weight.specialty) is not registered vocabulary")
        }
    }

    func testCapabilitiesAreExactlyTTS() {
        XCTAssertEqual(Set(Audio8MiniPackage.manifest.surfaces.map(\.capability)), [.tts])
    }

    func testProvenancePinsUpstreamRevision() {
        let provenance = Audio8MiniPackage.manifest.provenance
        XCTAssertEqual(provenance.sourceRepo, "Audio8/Audio8-TTS-Preview-0.1b")
        XCTAssertEqual(provenance.revision, "b476f0208438dfa791abee44d11029f055aeae04")
    }

    /// The surfaces must be distinguishable, or a consumer cannot select between them.
    func testSurfaceNameDiffersFromTheSibling() {
        let mini = Audio8MiniPackage.manifest.surfaces.map(\.name)
        let full = Audio8Package.manifest.surfaces.map(\.name)
        XCTAssertTrue(Set(mini).isDisjoint(with: Set(full)),
                      "surface names collide: \(mini) vs \(full)")
    }
}

final class MiniMaterializationConformanceTests: XCTestCase {

    /// MAT-1..5, against this package's own declarations.
    func testMaterializationGate() {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "audio8-mini-store-\(UUID().uuidString)")
        let report = MaterializationConformance.check(
            freshConfiguration: Audio8MiniConfiguration(modelsRootDirectory: root))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// The default configuration must point at THIS checkpoint. A default inherited from the
    /// sibling would load a different model than the manifest declares — the specific hazard
    /// that justifies `Audio8MiniConfiguration` existing as its own type.
    func testDefaultRepoIsTheCompactCheckpoint() {
        XCTAssertEqual(Audio8MiniConfiguration().repo,
                       "mlx-community/Audio8-TTS-Preview-0.1b-bf16")
        XCTAssertNotEqual(Audio8MiniConfiguration().repo, Audio8Configuration().repo)
    }

    func testExplicitDirectorySuppressesMaterialization() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "audio8-mini-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        FileManager.default.createFile(
            atPath: dir.appending(path: "codec.safetensors").path, contents: Data())
        let config = Audio8MiniConfiguration(modelDirectory: dir)
        XCTAssertTrue(config.missingWeightSources(storeRoot: nil).isEmpty)
    }

    func testEmptyStoreReportsSourceMissing() {
        let config = Audio8MiniConfiguration()
        let missing = config.missingWeightSources(
            storeRoot: FileManager.default.temporaryDirectory
                .appending(path: "audio8-mini-empty-\(UUID().uuidString)"))
        XCTAssertEqual(missing.map(\.role), ["main"])
    }
}

final class MiniCancellationConformanceTests: XCTestCase {

    /// CAN-1..3: a pre-cancelled run must surface `CancellationError` UNWRAPPED so the engine
    /// can tell user-cancel from governor-preempt.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = Audio8MiniPackage(configuration: Audio8MiniConfiguration())
        let task = Task {
            try await package.run(TTSRequest(text: "unreachable", voice: VoiceSelector(.auto)))
        }
        task.cancel()
        let result = await task.result
        switch result {
        case .success: XCTFail("a cancelled run must not produce a response")
        case .failure(let error):
            XCTAssertTrue(error is CancellationError,
                          "cancellation must arrive unwrapped, got \(type(of: error))")
        }
    }
}
