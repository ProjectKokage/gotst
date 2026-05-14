include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

set(GOTST_TEST_FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")

add_executable(gotst_unit_tests
    test_asr_frontend.cpp
    test_qwen3_forced_aligner.cpp
    test_speech_runtime_core.cpp
    test_speech_encoder_session.cpp
    test_speaker_mel.cpp
    test_speech_tokenizer.cpp
    test_tts_prompt_assembly.cpp
    test_tts_waveform_decoder.cpp
    test_ten_vad.cpp
    test_custom_voice.cpp
    test_cancellation.cpp
    test_language_config.cpp
    test_fft.cpp
    test_kernel.cpp
    test_gdscript_parity.cpp
    test_irodori_tts_session.cpp
)

target_link_libraries(gotst_unit_tests
    PRIVATE
        gotst-core
        Catch2::Catch2WithMain
)

set_target_properties(gotst_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(NOT MSVC)
    target_compile_options(gotst_unit_tests PRIVATE -frtti)
endif()

target_compile_definitions(gotst_unit_tests PRIVATE
    GOTST_TEST_FIXTURES_DIR="${GOTST_TEST_FIXTURES_DIR}"
)

include(Catch)
catch_discover_tests(gotst_unit_tests)
