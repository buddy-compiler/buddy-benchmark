import platform


def get_lib_extension() -> str:
    match platform.system():
        case "Linux":
            return "so"
        case "Darwin":
            return "dylib"
        case "Windows":
            return "dll"
        case _:
            raise Exception("Unknown platform")


def get_lib(path: str) -> str:
    return f"{path}.{get_lib_extension()}"
