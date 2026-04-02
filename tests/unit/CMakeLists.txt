add_executable(gotst_tests
    test_speech_runtime_core.cpp
)

target_link_libraries(gotst_tests
    PRIVATE
        gotst-core
)

set_target_properties(gotst_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME gotst_tests COMMAND gotst_tests)
