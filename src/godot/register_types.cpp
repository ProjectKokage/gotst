#include "gotst/godot/speech_runtime.hpp"
#include "gotst/godot/speech_runtime_config.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

namespace {

void initialize_gotst_module(ModuleInitializationLevel level) {
    if(level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    ClassDB::register_class<GotstSpeechRuntimeConfig>();
    ClassDB::register_class<GotstSpeechRuntime>();
}

void uninitialize_gotst_module(ModuleInitializationLevel level) {
    if(level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

} // namespace

extern "C" {

GDExtensionBool GDE_EXPORT gotst_library_init(
    GDExtensionInterfaceGetProcAddress get_proc_address,
    GDExtensionClassLibraryPtr library,
    GDExtensionInitialization *initialization
) {
    GDExtensionBinding::InitObject init_object(get_proc_address, library, initialization);
    init_object.register_initializer(initialize_gotst_module);
    init_object.register_terminator(uninitialize_gotst_module);
    init_object.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_object.init();
}

}
