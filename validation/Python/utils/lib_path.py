import platform

def format_lib_path(lib_name:str):
    if platform.system() == "Linux":
        lib_name += ".so"
    elif platform.system() == "Darwin":
        lib_name += ".dylib"
    return lib_name
