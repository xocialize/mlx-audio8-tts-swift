// Conformance for `Audio8RealisticPackage` (Audio8-TTS-Realistic-0.6b).
//
// The Realistic and Preview checkpoints share every tensor name and shape, so most of what
// could go wrong here is silent: a default repo that points at the sibling, a provenance copied
// from it, or a surface name that collides with it would all load, run and pass a live gate.
// These tests pin the declarations that tell the two apart.

import Foundation
import MLXToolKit
import MLXServeConformance
import MLXServeCore
import XCTest

@testable import MLXAudio8TTS

final class RealisticManifestConformanceTests: XCTestCase {

    /// C7/C8 — single permissive layer, like the Preview: the LM is Apache-2.0 and the codec is
    /// the Preview's, byte for byte.
    func testLicenseIsPermissiveOnBothLayers() {
        let license = Audio8RealisticPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertEqual(license.portCodeLicense, .apache2)
        XCTAssertTrue(SPDXLicense.permissiveAllowlist.contains(license.weightLicense))
    }

    /// Same tensor shapes as the Preview, so the same bound: a resident floor under the weight
    /// bytes (~1.2 GB bf16 LM + ~1.35 GB fp32 codec) could not have come from a real load.
    func testFootprintIsSplitAndMeasured() {
        guard let footprint = Audio8RealisticPackage.manifest.requirements.footprints
            .first(where: { $0.quant == .bf16 }) else {
            return XCTFail("no bf16 footprint declared")
        }
        XCTAssertGreaterThan(footprint.residentBytes, 2_500_000_000,
                             "resident floor is below the weight bytes — was it measured?")
        // The gates print MiB. Compare in bytes, or a MiB figure read as decimal MB slips under
        // the measurement it came from (AB-L-0116).
        XCTAssertGreaterThan(footprint.residentBytes, 2434 << 20,
                             "declared resident is below the measured 2434 MiB floor")
        XCTAssertGreaterThan(footprint.peakActivationBytes, 3658 << 20,
                             "declared activation is below the measured 3658 MiB batch transient")
    }

    /// Speaker similarity measured tied with the Preview (0.7211 vs 0.7212), so the selection
    /// signal must be tied too — neither weaker (it is not) nor stronger (the card's gain did
    /// not reproduce).
    func testVoiceCloneClaimEqualsTheSibling() {
        func strength(_ specialties: [SpecialtyWeight]) -> Double? {
            specialties.first { $0.specialty == .voiceClone }?.strength
        }
        XCTAssertEqual(strength(Audio8RealisticPackage.manifest.specialties),
                       strength(Audio8Package.manifest.specialties))
    }

    func testSpecialtiesAreRegistered() {
        for weight in Audio8RealisticPackage.manifest.specialties {
            XCTAssertTrue(weight.specialty.isRegistered,
                          "\(weight.specialty) is not registered vocabulary")
        }
    }

    func testCapabilitiesAreExactlyTTS() {
        XCTAssertEqual(Set(Audio8RealisticPackage.manifest.surfaces.map(\.capability)), [.tts])
    }

    /// A different upstream repo from a different publisher (Edge0, not Audio8) — the one place
    /// a copy-paste from the Preview's manifest would be wrong and still compile.
    func testProvenancePinsUpstreamRevision() {
        let provenance = Audio8RealisticPackage.manifest.provenance
        XCTAssertEqual(provenance.sourceRepo, "Edge0/Audio8-TTS-Realistic-Air")
        XCTAssertEqual(provenance.revision, "925bbd4c2d8360af85f2594bf0205173fda853ba")
    }

    func testSurfaceNameDiffersFromBothSiblings() {
        let realistic = Set(Audio8RealisticPackage.manifest.surfaces.map(\.name))
        let preview = Set(Audio8Package.manifest.surfaces.map(\.name))
        let mini = Set(Audio8MiniPackage.manifest.surfaces.map(\.name))
        XCTAssertTrue(realistic.isDisjoint(with: preview.union(mini)),
                      "surface names collide: \(realistic) vs \(preview) / \(mini)")
    }

    /// The languages the V0 A/B declined to declare must not leak into the surface as supported.
    func testUndeclaredLanguagesAreMarkedUnsupported() {
        guard let summary = Audio8RealisticPackage.manifest.surfaces.first?.summary else {
            return XCTFail("no surface")
        }
        XCTAssertTrue(summary.contains("Not supported: Arabic, Thai, Vietnamese"), summary)
    }

    func testStreamingAdvertisementIsCoherent() {
        let report = StreamingConformance.checkAdvertisement(Audio8RealisticPackage.self)
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RealisticMaterializationConformanceTests: XCTestCase {

    func testMaterializationGate() {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "audio8-realistic-store-\(UUID().uuidString)")
        let report = MaterializationConformance.check(
            freshConfiguration: Audio8RealisticConfiguration(modelsRootDirectory: root))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// The hazard this configuration type exists for: the two 0.6b bundles are interchangeable
    /// to the loader, so only the default repo keeps the package honest about what it loads.
    func testDefaultRepoIsTheRealisticCheckpoint() {
        XCTAssertEqual(Audio8RealisticConfiguration().repo,
                       "mlx-community/Audio8-TTS-Realistic-0.6b-bf16")
        XCTAssertNotEqual(Audio8RealisticConfiguration().repo, Audio8Configuration().repo)
    }

    func testEmptyStoreReportsSourceMissing() {
        let config = Audio8RealisticConfiguration()
        let missing = config.missingWeightSources(
            storeRoot: FileManager.default.temporaryDirectory
                .appending(path: "audio8-realistic-empty-\(UUID().uuidString)"))
        XCTAssertEqual(missing.map(\.role), ["main"])
    }
}

final class RealisticCancellationConformanceTests: XCTestCase {

    func testPreCancelledRunPropagatesCancellation() async {
        let package = Audio8RealisticPackage(configuration: Audio8RealisticConfiguration())
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
