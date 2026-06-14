cmake_minimum_required(VERSION 3.28)

project(
    gotst
    VERSION 0.1.0
    DESCRIPTION "Speech-focused Godot GDExtension built on godorama and gonx cores"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GOTST_BUILD_GDEXTENSION "Build the Godot-facing GDExtension shared library" ON)
option(GOTST_BUILD_TESTS "Build native tests" ON)
option(GOTST_BUILD_CLI "Build the standalone gotst command-line tool" OFF)

set(
    GOTST_GODORAMA_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../godorama"
    CACHE PATH
    "Path to the godorama repository"
)
set(
    GOTST_GONX_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../gonx"
    CACHE PATH
    "Path to the gonx repository"
)
set(
    GOTST_GODOT_CPP_SOURCE_DIR
    "${GOTST_GODORAMA_SOURCE_DIR}/thirdparty/godot-cpp"
    CACHE PATH
    "Path to a shared godot-cpp source tree"
)

function(gotst_require_dir dir_path description)
    if(NOT EXISTS "${dir_path}")
        message(FATAL_ERROR "${description} not found at ${dir_path}")
    endif()
endfunction()

gotst_require_dir("${GOTST_GODORAMA_SOURCE_DIR}" "godorama source directory")
gotst_require_dir("${GOTST_GONX_SOURCE_DIR}" "gonx source directory")

if(GOTST_BUILD_GDEXTENSION)
    if(TARGET godot-cpp)
        set(GOTST_GODOT_CPP_TARGET godot-cpp)
        set(GOTST_GODOT_CPP_PROPERTY_TARGET godot-cpp)
    elseif(TARGET godot::cpp)
        set(GOTST_GODOT_CPP_TARGET godot::cpp)
        set(GOTST_GODOT_CPP_PROPERTY_TARGET godot::cpp)
    else()
        gotst_require_dir("${GOTST_GODOT_CPP_SOURCE_DIR}" "godot-cpp source directory")
        add_subdirectory("${GOTST_GODOT_CPP_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/godot-cpp" EXCLUDE_FROM_ALL)
        if(TARGET godot-cpp)
            set(GOTST_GODOT_CPP_TARGET godot-cpp)
            set(GOTST_GODOT_CPP_PROPERTY_TARGET godot-cpp)
        elseif(TARGET godot::cpp)
            set(GOTST_GODOT_CPP_TARGET godot::cpp)
            set(GOTST_GODOT_CPP_PROPERTY_TARGET godot::cpp)
        else()
            message(FATAL_ERROR "godot-cpp target was not created by ${GOTST_GODOT_CPP_SOURCE_DIR}")
        endif()
    endif()
endif()

if(NOT TARGET gonx-core AND NOT TARGET gonx::core)
    set(GONX_BUILD_GDEXTENSION OFF CACHE BOOL "" FORCE)
    set(GONX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GOTST_GONX_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/gonx" EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET godot-llama-core AND NOT TARGET godot_llama::core)
    set(GODOT_LLAMA_BUILD_GDEXTENSION OFF CACHE BOOL "" FORCE)
    set(GODOT_LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GODORAMA_BUILD_TOKENIZERS ON CACHE BOOL "" FORCE)
    add_subdirectory("${GOTST_GODORAMA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/godorama" EXCLUDE_FROM_ALL)
endif()

if(TARGET gonx::core)
    set(GOTST_GONX_CORE_TARGET gonx::core)
else()
    set(GOTST_GONX_CORE_TARGET gonx-core)
endif()

if(TARGET godot_llama::core)
    set(GOTST_GODORAMA_CORE_TARGET godot_llama::core)
else()
    set(GOTST_GODORAMA_CORE_TARGET godot-llama-core)
endif()

if(TARGET godot_llama::tokenizers)
    set(GOTST_TOKENIZERS_TARGET godot_llama::tokenizers)
elseif(TARGET tokenizers_cpp)
    set(GOTST_TOKENIZERS_TARGET tokenizers_cpp)
else()
    message(FATAL_ERROR "gotst text tokenization requires GODORAMA_BUILD_TOKENIZERS=ON to provide tokenizers_cpp")
endif()

add_library(gotst-ten-vad STATIC
    thirdparty/ten_vad/src/aed.cc
    thirdparty/ten_vad/src/biquad.cc
    thirdparty/ten_vad/src/fftw.c
    thirdparty/ten_vad/src/fscvrt.cc
    thirdparty/ten_vad/src/pitch_est.cc
    thirdparty/ten_vad/src/stft.cc
)
add_library(gotst::ten_vad ALIAS gotst-ten-vad)

target_include_directories(gotst-ten-vad
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ten_vad/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ten_vad/include"
)

target_link_libraries(gotst-ten-vad
    PUBLIC
        onnxruntime::onnxruntime
)

set_target_properties(gotst-ten-vad PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(NOT MSVC)
    target_compile_options(gotst-ten-vad PRIVATE -Wno-unused-parameter -Wno-sign-compare -Wno-write-strings)
endif()

add_library(gotst-core STATIC
    src/core/speech_runtime_core.cpp
    src/core/speech_encoder_session.cpp
    src/core/asr_frontend.cpp
    src/core/speaker_mel.cpp
    src/core/tts_prompt_assembly.cpp
    src/core/tts_code_generator.cpp
    src/core/tts_waveform_decoder.cpp
    src/core/qwen_tts_pipeline.cpp
    src/core/irodori_tts_session.cpp
    src/core/text_tokenization.cpp
    src/core/asr_token_decoder.cpp
    src/core/qwen3_forced_aligner.cpp
    src/core/ten_vad.cpp
    src/core/language_config.cpp
    src/core/fft.cpp
)
add_library(gotst::core ALIAS gotst-core)

target_include_directories(gotst-core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(gotst-core
    PUBLIC
        ${GOTST_GONX_CORE_TARGET}
        ${GOTST_GODORAMA_CORE_TARGET}
        ${GOTST_TOKENIZERS_TARGET}
        gotst-ten-vad
)

set_target_properties(gotst-core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(NOT MSVC)
    target_compile_options(gotst-core PRIVATE -Wall -Wextra)
endif()

if(GOTST_BUILD_GDEXTENSION)
    add_library(gotst SHARED
        src/godot/register_types.cpp
        src/godot/speech_runtime.cpp
        src/godot/speech_runtime_config.cpp
    )
    add_library(gotst::godot ALIAS gotst)

    target_include_directories(gotst
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    target_link_libraries(gotst
        PRIVATE
            gotst-core
            ${GOTST_GODOT_CPP_TARGET}
    )

    get_target_property(GODOTCPP_SUFFIX ${GOTST_GODOT_CPP_PROPERTY_TARGET} GODOTCPP_SUFFIX)
    if(NOT GODOTCPP_SUFFIX)
        set(GODOTCPP_SUFFIX "")
    endif()

    set_target_properties(gotst PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_VISIBILITY_PRESET hidden
        PREFIX "lib"
        OUTPUT_NAME "gotst${GODOTCPP_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/addons/gotst/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/addons/gotst/bin"
    )

    if(NOT MSVC)
        target_compile_options(gotst PRIVATE -Wall -Wextra)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_target_properties(gotst PROPERTIES
            BUILD_RPATH "\$ORIGIN"
            INSTALL_RPATH "\$ORIGIN"
            BUILD_WITH_INSTALL_RPATH ON
            BUILD_RPATH_USE_ORIGIN ON
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set_target_properties(gotst PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH ON
            SUFFIX ".dylib"
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set_target_properties(gotst PROPERTIES
            PREFIX ""
            SUFFIX ".dll"
        )
    endif()

    get_target_property(_ort_location onnxruntime::onnxruntime IMPORTED_LOCATION)
    if(_ort_location)
        add_custom_command(TARGET gotst POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ort_location}"
                "$<TARGET_FILE_DIR:gotst>"
            COMMENT "Copying ONNX Runtime shared library to the gotst output directory"
        )

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            get_filename_component(_ort_dir "${_ort_location}" DIRECTORY)
            file(GLOB _ort_symlinks "${_ort_dir}/libonnxruntime.so*")
            foreach(_sym ${_ort_symlinks})
                add_custom_command(TARGET gotst POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_sym}"
                        "$<TARGET_FILE_DIR:gotst>"
                    COMMENT "Copying ${_sym}"
                )
            endforeach()
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            get_filename_component(_ort_dir "${_ort_location}" DIRECTORY)
            file(GLOB _ort_symlinks LIST_DIRECTORIES false "${_ort_dir}/libonnxruntime*.dylib")
            foreach(_sym ${_ort_symlinks})
                add_custom_command(TARGET gotst POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_sym}"
                        "$<TARGET_FILE_DIR:gotst>"
                    COMMENT "Copying ${_sym}"
                )
            endforeach()
        endif()
    endif()

    install(TARGETS gotst
        LIBRARY DESTINATION "addons/gotst/bin"
        RUNTIME DESTINATION "addons/gotst/bin"
    )

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/addons/gotst/gotst.gdextension"
        DESTINATION "addons/gotst"
    )

    if(_ort_location)
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            install(FILES "${_ort_location}" DESTINATION "addons/gotst/bin")
        else()
            get_filename_component(_ort_dir "${_ort_location}" DIRECTORY)
            file(GLOB _ort_libs LIST_DIRECTORIES false "${_ort_dir}/libonnxruntime*")
            install(FILES ${_ort_libs} DESTINATION "addons/gotst/bin")
        endif()
    endif()
endif()

if(GOTST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/unit)
endif()

if(GOTST_BUILD_CLI)
    add_subdirectory(cli)
endif()
