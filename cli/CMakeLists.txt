add_executable(gotst-cli
    main.cpp
    cli_common.cpp
    cmd_inspect.cpp
    cmd_irodori_tts.cpp
    cmd_qwen_tts.cpp
)

target_include_directories(gotst-cli
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_link_libraries(gotst-cli
    PRIVATE
        gotst-core
)

set_target_properties(gotst-cli PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "gotst"
)

if(NOT MSVC)
    target_compile_options(gotst-cli PRIVATE -Wall -Wextra)
endif()
