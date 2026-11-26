def test_streaming_runtime_types_are_public_exports() -> None:
    from nhsmm import HSMMFilterRuntime, HSMMFilterState, HSMMRuntimeState

    assert HSMMFilterRuntime.__name__ == "HSMMFilterRuntime"
    assert HSMMFilterState.__name__ == "HSMMFilterState"
    assert HSMMRuntimeState.__name__ == "HSMMRuntimeState"
