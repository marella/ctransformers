from typing import Optional
import platform
from pathlib import Path


def find_library(path: Optional[str] = None, cuda: bool = False) -> str:
    lib_directory = Path(__file__).parent.resolve() / "lib"

    if path:
        subdirs = [d.name for d in lib_directory.iterdir() if d.is_dir()]
        if path not in subdirs:
            return path

    system = platform.system()
    if not path:
        if (lib_directory / "local").is_dir():
            path = "local"
        elif cuda:
            path = "cuda"
        else:
            from cpuinfo import get_cpu_info

            flags = get_cpu_info()["flags"]

            if "avx2" in flags:
                path = "avx2"
            elif "avx" in flags and "f16c" in flags:
                path = "avx"
            else:
                path = "basic"

    name = "ctransformers"
    if system == "Linux":
        name = f"lib{name}.so"
    elif system == "Windows":
        name = f"{name}.dll"
    elif system == "Darwin":
        name = f"lib{name}.dylib"
    else:
        name = ""

    path = lib_directory / path / name
    if not path.is_file():
        raise OSError(
            "Precompiled binaries are not available for the current platform. "
            "Please reinstall from source using:\n\n"
            "  pip uninstall ctransformers --yes\n"
            f"  {'CT_CUBLAS=1 ' if cuda else ''}pip install ctransformers --no-binary ctransformers\n\n"
        )
    return str(path)
