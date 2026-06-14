#include "cli_common.hpp"

#include <filesystem>
#include <iostream>

namespace gotst_cli {

namespace {

void print_artifact(const std::filesystem::path &bundle_root, const JsonValue *artifacts, const std::string &key) {
    const std::filesystem::path path = resolve_path(bundle_root, json_string(artifacts, key));
    if(path.empty()) {
        std::cout << key << ": <missing>\n";
        return;
    }
    std::cout << key << ": " << path.string()
              << (std::filesystem::exists(path) ? " [ok]" : " [missing]") << '\n';
}

} // namespace

int command_inspect(const ParsedArgs &args) {
    if(args.has("help") || args.has("h")) {
        std::cout
            << "Usage: gotst inspect --bundle DIR [--manifest irodori_bundle.json]\n";
        return 0;
    }

    const std::filesystem::path bundle_root = args.value("bundle");
    if(bundle_root.empty()) {
        return print_error("inspect requires --bundle DIR", 2);
    }
    const std::filesystem::path manifest_path = resolve_path(bundle_root, args.value("manifest", "irodori_bundle.json"));
    auto manifest = read_json_file(manifest_path);
    if(!manifest.is_ok()) {
        return print_error(manifest.get_error());
    }
    const JsonValue *artifacts = manifest.value().get("artifacts");

    std::cout << "manifest: " << manifest_path.string() << '\n';
    std::cout << "mode: " << json_string(&manifest.value(), "mode", "<unknown>") << '\n';
    print_artifact(bundle_root, artifacts, "tokenizer_json_path");
    print_artifact(bundle_root, artifacts, "tokenizer_config_path");
    print_artifact(bundle_root, artifacts, "text_encoder_onnx_path");
    print_artifact(bundle_root, artifacts, "caption_encoder_onnx_path");
    print_artifact(bundle_root, artifacts, "speaker_encoder_onnx_path");
    print_artifact(bundle_root, artifacts, "duration_predictor_onnx_path");
    print_artifact(bundle_root, artifacts, "dit_step_onnx_path");
    print_artifact(bundle_root, artifacts, "dacvae_encoder_onnx_path");
    print_artifact(bundle_root, artifacts, "dacvae_decoder_onnx_path");
    return 0;
}

} // namespace gotst_cli
