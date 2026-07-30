import XCTest

@testable import MLXAudio8TTS

final class PlaceholderTests: XCTestCase {
    func testVersion() {
        XCTAssertFalse(MLXAudio8TTS.version.isEmpty)
    }
}
