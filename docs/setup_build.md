# gotst Setup and Build Guide

This guide describes how to build `gotst` as a standalone speech runtime
repository. It assumes `gotst`, `godorama`, and `gonx` are separate source
checkouts, but the same CMake options also work from a parent superbuild.

`gotst` builds on:

- `godorama` for generic `llama.cpp` and tokenizer-backed text support
- `gonx` for generic ONNX Runtime support
- `godot-cpp` for the optional Godot GDExtension boundary

`gotst` itself should stay focused on speech-specific runtime logic.

## Prerequisites

Install these tools before configuring the build:

- Git
- CMake 3.28 or newer
- Ninja, recommended for consistent command examples
- A C++20 compiler
- Rust/Cargo when the selected `godorama` tokenizer build needs the
  HuggingFace tokenizer bridge
- Godot 4-compatible `godot-cpp`, normally supplied through the `godorama`
  checkout

Platform notes:

- macOS: install Xcode Command Line Tools with `xcode-select --install`.
- Linux: install the usual compiler toolchain, CMake, Ninja, Git, and Rust
  packages for your distribution.
- ONNX Runtime is supplied through the `gonx` build. If you already have an
  ONNX Runtime package, pass it through the `GONX_ORT_ROOT` CMake option.

## Source Layout

The default CMake paths expect this sibling layout:

```text
workspace/
  gotst/
  godorama/
  gonx/
```

Initialize each checkout's submodules before building. The exact submodule set
is owned by each dependency, so recursive initialization is the safest fresh
clone setup:

```bash
git -C gotst submodule update --init --recursive
git -C godorama submodule update --init --recursive
git -C gonx submodule update --init --recursive
```

For a different layout, pass explicit paths at configure time:

```bash
-DGOTST_GODORAMA_SOURCE_DIR=/absolute/path/to/godorama
-DGOTST_GONX_SOURCE_DIR=/absolute/path/to/gonx
-DGOTST_GODOT_CPP_SOURCE_DIR=/absolute/path/to/godot-cpp
```

`GOTST_GODOT_CPP_SOURCE_DIR` defaults to
`<godorama>/thirdparty/godot-cpp`.

## Build The Godot Addon

Use a Debug build for Godot editor runs:

```bash
cmake -S gotst -B gotst/build/debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGOTST_BUILD_GDEXTENSION=ON \
  -DGOTST_BUILD_TESTS=OFF \
  -DGODOTCPP_TARGET=template_debug \
  -DGOTST_GODORAMA_SOURCE_DIR="$PWD/godorama" \
  -DGOTST_GONX_SOURCE_DIR="$PWD/gonx"

cmake --build gotst/build/debug --parallel
```

Use a Release build for export templates or release validation:

```bash
cmake -S gotst -B gotst/build/release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGOTST_BUILD_GDEXTENSION=ON \
  -DGOTST_BUILD_TESTS=OFF \
  -DGODOTCPP_TARGET=template_release \
  -DGOTST_GODORAMA_SOURCE_DIR="$PWD/godorama" \
  -DGOTST_GONX_SOURCE_DIR="$PWD/gonx"

cmake --build gotst/build/release --parallel
```

The addon output is written inside the `gotst` checkout:

```text
gotst/addons/gotst/gotst.gdextension
gotst/addons/gotst/bin/
```

The built `gotst` shared library and the matching ONNX Runtime shared library
are placed under `addons/gotst/bin/`. To consume the addon from a Godot project,
copy or symlink `gotst/addons/gotst` to the project's `addons/gotst` directory.

## Build Native Tests

The native tests validate speech core behavior without loading the Godot
extension. They still need `godorama` and `gonx` because `gotst-core` links
against their core targets.

```bash
cmake -S gotst -B gotst/build/test -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGOTST_BUILD_GDEXTENSION=OFF \
  -DGOTST_BUILD_TESTS=ON \
  -DGOTST_BUILD_CLI=OFF \
  -DGOTST_GODORAMA_SOURCE_DIR="$PWD/godorama" \
  -DGOTST_GONX_SOURCE_DIR="$PWD/gonx"

cmake --build gotst/build/test --parallel
ctest --test-dir gotst/build/test --output-on-failure
```

The test build fetches Catch2 through CMake `FetchContent` unless it is already
available in CMake's dependency cache. Offline environments should pre-populate
that cache or provide Catch2 through the normal CMake override mechanisms.

## Build The CLI

The standalone CLI is useful for inspecting bundles and running speech synthesis
without Godot:

```bash
cmake -S gotst -B gotst/build/cli -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGOTST_BUILD_GDEXTENSION=OFF \
  -DGOTST_BUILD_TESTS=OFF \
  -DGOTST_BUILD_CLI=ON \
  -DGOTST_GODORAMA_SOURCE_DIR="$PWD/godorama" \
  -DGOTST_GONX_SOURCE_DIR="$PWD/gonx"

cmake --build gotst/build/cli --parallel
gotst/build/cli/cli/gotst --help
```

See [`../cli/README.md`](../cli/README.md) for command examples.

## Common CMake Options

`gotst` options:

- `GOTST_BUILD_GDEXTENSION`: build the Godot-facing shared library.
- `GOTST_BUILD_TESTS`: build native unit tests.
- `GOTST_BUILD_CLI`: build the standalone `gotst` executable.
- `GOTST_GODORAMA_SOURCE_DIR`: path to the `godorama` source checkout.
- `GOTST_GONX_SOURCE_DIR`: path to the `gonx` source checkout.
- `GOTST_GODOT_CPP_SOURCE_DIR`: path to `godot-cpp` when not using the default
  under `godorama`.

Useful dependency options passed through to child projects:

- `GONX_ORT_ROOT`: path to an existing ONNX Runtime package.
- `GONX_ORT_VARIANT`: select the ONNX Runtime package variant when supported by
  `gonx`.
- `GONX_ORT_FROM_SOURCE`: ask `gonx` to build ONNX Runtime from source when
  supported.
- `GONX_ORT_PROVIDERS`: select ONNX Runtime execution providers when supported.
- `GGML_METAL`, `GGML_CUDA`, `GGML_VULKAN`: select the `llama.cpp` backend
  through the embedded `godorama` build.

## Parent Superbuilds

A parent CMake project can add `gotst` with `add_subdirectory()` and predefine
dependency targets. `gotst` reuses these targets when they already exist:

- `godot-cpp` or `godot::cpp`
- `gonx-core` or `gonx::core`
- `godot-llama-core` or `godot_llama::core`
- `tokenizers_cpp` or `godot_llama::tokenizers`

When those targets are not already present, `gotst` adds the configured
`godorama`, `gonx`, and `godot-cpp` source directories as child builds.

## Troubleshooting

`godorama source directory not found` or `gonx source directory not found`:

- Check the sibling checkout layout, or pass `GOTST_GODORAMA_SOURCE_DIR` and
  `GOTST_GONX_SOURCE_DIR` explicitly.

`godot-cpp source directory not found`:

- Initialize `godorama` submodules, or pass `GOTST_GODOT_CPP_SOURCE_DIR` to a
  compatible Godot 4 `godot-cpp` checkout.

`gotst text tokenization requires GODORAMA_BUILD_TOKENIZERS=ON`:

- Ensure the `godorama` checkout includes its tokenizer dependencies and that no
  parent build disabled its tokenizer target.

ONNX Runtime target or library errors:

- Build/configure `gonx` with a valid ONNX Runtime package, or pass
  `GONX_ORT_ROOT` to an existing package.
- For Godot addon use, keep the ONNX Runtime shared library beside the built
  `gotst` library in `addons/gotst/bin/`.

Godot cannot load `gotst.gdextension`:

- Build the matching Debug or Release template target for the running Godot
  mode.
- Confirm the platform library name in `addons/gotst/gotst.gdextension`
  matches the file emitted in `addons/gotst/bin/`.
- Confirm ONNX Runtime is present in `addons/gotst/bin/`.
- On Linux and macOS, keep the addon bin directory layout intact so `$ORIGIN`
  or `@loader_path` RPATH lookup can find companion libraries.

Tests cannot fetch Catch2:

- Pre-populate CMake's `FetchContent` cache, use a network-enabled configure
  environment, or build with `GOTST_BUILD_TESTS=OFF`.

## Generated Outputs

The normal generated paths are:

```text
build/
addons/gotst/bin/
compile_commands.json
```

Do not commit generated binaries or build trees. Commit source, headers,
fixtures, and documentation.
