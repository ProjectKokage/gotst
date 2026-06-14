#include "cli_common.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    try {
        if(argc < 2) {
            gotst_cli::print_global_help();
            return 2;
        }

        const std::string command(argv[1]);
        if(command == "help" || command == "--help" || command == "-h") {
            gotst_cli::print_global_help();
            return 0;
        }

        auto args = gotst_cli::ParsedArgs::parse(argc, argv, 2);
        if(!args.is_ok()) {
            return gotst_cli::print_error(args.get_error());
        }

        if(command == "inspect") {
            return gotst_cli::command_inspect(args.value());
        }
        if(command == "irodori-tts") {
            return gotst_cli::command_irodori_tts(args.value());
        }
        if(command == "qwen-tts") {
            return gotst_cli::command_qwen_tts(args.value());
        }

        std::cerr << "error: unknown command: " << command << "\n\n";
        gotst_cli::print_global_help();
        return 2;
    } catch(const std::exception &ex) {
        std::cerr << "error: unhandled exception: " << ex.what() << '\n';
        return 1;
    } catch(...) {
        std::cerr << "error: unhandled non-standard exception\n";
        return 1;
    }
}
