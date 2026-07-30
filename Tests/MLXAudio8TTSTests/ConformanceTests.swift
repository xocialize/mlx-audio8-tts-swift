// ConformanceTests.swift — Audio8-TTS through the engine's offline gates.
//
// These run without weights and without a kernel: manifest/declaration checks (C0–C13), the
// materialization gate (MAT-1..5), the cancellation gate (CAN-1..3), and the inference-mode
// gate (C14/INF). The live checks — a real load, a real run, the measured footprint — are the
// `audio8-gates --validate` CLI lane and the app harness, because the SPM test product's
// metallib is unreliable for GPU work.

import Foundation
import MLXToolKit
import MLXServeConformance
import XCTest

@testable import MLXAudio8TTS

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 — both layers permissive, so the package admits under the default
    /// `.permissiveOnly` policy with no acknowledgement flow. The upstream ships the LM and the
    /// codec in ONE Apache-2.0 repo, which is why this is a single-layer declaration rather than
    /// Gepard's more-restrictive-of-two.
    func testLicenseIsPermissiveOnBothLayers() {
        let license = Audio8Package.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertEqual(license.portCodeLicense, .apache2)
        XCTAssertTrue(SPDXLicense.permissiveAllowlist.contains(license.weightLicense),
                      "weight license must be on the permissive allowlist to admit by default")
    }

    /// C-memory — the footprint is declared SPLIT (a resident floor plus a separate activation
    /// peak), not one flat number that bakes the transient into residency.
    func testFootprintIsSplitAndNonZero() {
        guard let footprint = Audio8Package.manifest.requirements.footprints
            .first(where: { $0.quant == .bf16 }) else {
            return XCTFail("no bf16 footprint declared")
        }
        XCTAssertGreaterThan(footprint.residentBytes, 0)
        XCTAssertGreaterThan(footprint.peakActivationBytes, 0,
                             "activation peak must be declared separately from the resident floor")
        // The weights alone are ~1.2 GB (bf16 LM) + ~1.35 GB (fp32 codec); a resident floor below
        // that would mean the declaration was never measured.
        XCTAssertGreaterThan(footprint.residentBytes, 2_500_000_000,
                             "resident floor is below the weight bytes — was it measured?")
    }

    /// C6 — every declared specialty is in the engine's registered vocabulary (the check that
    /// catches invented terms like a would-be "multilingual").
    func testSpecialtiesAreRegistered() {
        for weight in Audio8Package.manifest.specialties {
            XCTAssertTrue(weight.specialty.isRegistered,
                          "unregistered specialty \(weight.specialty.rawValue) — add it to the "
                          + "engine vocabulary in the same change, or drop it")
        }
    }

    /// The manifest must not claim streaming: this package decodes the whole utterance in one
    /// pass and does not conform to `StreamEmitting`. A `.audioChunk` declaration here would be
    /// a false capability claim to the orchestrator.
    func testDoesNotClaimStreaming() {
        XCTAssertFalse(Audio8Package.manifest.specialties.contains { $0.specialty == .realtimeStreaming },
                       "Audio8 synthesizes whole utterances — do not claim realtimeStreaming")
        for surface in Audio8Package.manifest.surfaces {
            XCTAssertNil(surface.streaming,
                         "surface \(surface.name) declares streaming but the package is not StreamEmitting")
        }
    }

    /// C1 — the package serves the `tts` capability and nothing it cannot do.
    func testCapabilitiesAreExactlyTTS() {
        XCTAssertEqual(Set(Audio8Package.manifest.capabilities), [.tts])
    }

    /// Provenance pins the exact upstream revision the port was validated against (the sha the
    /// oracle goldens were captured from).
    func testProvenancePinsUpstreamRevision() {
        let provenance = Audio8Package.manifest.provenance
        XCTAssertEqual(provenance.sourceRepo, "Audio8/Audio8-TTS-Preview-0.6b")
        XCTAssertEqual(provenance.revision, "1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17")
    }
}

final class MaterializationConformanceTests: XCTestCase {

    /// MAT-1..5 on a fresh (dir-less) configuration: the engine can stamp the store root, the
    /// sources are declared and well-formed, and a config with nothing on disk reports itself
    /// as needing materialization.
    func testMaterializationGate() {
        let report = MaterializationConformance.check(
            freshConfiguration: Audio8Configuration(modelsRootDirectory: emptyStoreRoot()))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// An explicit `modelDirectory` is the dev escape hatch: when the probe file is present the
    /// package must report nothing missing, so `load()` never touches the network.
    func testExplicitDirectorySuppressesMaterialization() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio8-explicit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data().write(to: directory.appending(path: Audio8Configuration.mainProbeFile))

        let configuration = Audio8Configuration(modelDirectory: directory)
        XCTAssertTrue(configuration.missingWeightSources(storeRoot: nil).isEmpty,
                      "an explicit directory holding the probe file must suppress materialization")
    }

    /// A dir-less configuration with an empty store reports its single source as missing.
    func testEmptyStoreReportsSourceMissing() {
        let configuration = Audio8Configuration(modelsRootDirectory: emptyStoreRoot())
        let missing = configuration.missingWeightSources(storeRoot: configuration.modelsRootDirectory)
        XCTAssertEqual(missing.map(\.role), ["main"])
    }

    /// The declared file list must cover everything `load()` opens — weights, config, and BOTH
    /// tokenizer files. A missing entry here is the classic "works on my machine, 404s on a
    /// fresh install" bug.
    func testDeclaredFilesCoverEverythingLoadOpens() {
        let declared = Set(Audio8Configuration.mainFiles)
        for required in ["model.safetensors", "codec.safetensors", "config.json",
                         "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(declared.contains(required),
                          "\(required) is opened at load but not declared as a weight-source file")
        }
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("audio8-empty-store-\(UUID().uuidString)")
    }
}

final class CancellationConformanceTests: XCTestCase {

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` UNCHANGED (never
    /// wrapped in `Audio8PackageError`), which is how the engine tells a user cancel from a
    /// governor preempt. Note this passes on an UNLOADED package by design: the entry
    /// checkpoint precedes the `notLoaded` validation.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = Audio8Package(configuration: Audio8Configuration())
        let request = TTSRequest(text: "cancellation probe")
        let report = await CancellationConformance.checkRun(package: package, request: request)
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — the declared checkpoint cadence. Audio8's run is long (autoregressive rollout at
    /// 21.5 frames/s plus a whole-utterance codec decode), so `.subSecondRuns` would be false;
    /// the rollout checkpoints once per generated frame and reports `RunProgress` on the same
    /// seam, which is the observable evidence for the claim.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: Audio8Package.manifest,
            posture: .cadence([
                .init(phase: .generate, unit: .frame, reportsRunProgress: true),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
