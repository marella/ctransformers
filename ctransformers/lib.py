import platform
from ctypes import CDLL
from typing import List, Optional
from pathlib import Path

from .logger import logger


def find_library(path: Optional[str] = None, gpu: bool = False) -> str:
    lib_directory = Path(__file__).parent.resolve() / "lib"

    if path:
        subdirs = [d.name for d in lib_directory.iterdir() if d.is_dir()]
        if path not in subdirs:
            return path

    system = platform.system()
    metal = gpu and system == "Darwin"
    cuda = gpu and not metal
    if not path:
        if (lib_directory / "local").is_dir():
            path = "local"
        elif cuda:
            path = "cuda"
        elif metal:
            path = "metal"
        elif platform.processor() == "arm":
            # Apple silicon doesn't support AVX/AVX2.
            path = "basic" if system == "Darwin" else ""
        else:
            from cpuinfo import get_cpu_info

            try:
                flags = get_cpu_info()["flags"]
            except:
                logger.warning(
                    "Unable to detect CPU features. "
                    "Please report at https://github.com/marella/ctransformers/issues"
                )
                flags = []

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
        if cuda:
            env = "CT_CUBLAS=1 "
        elif metal:
            env = "CT_METAL=1 "
        else:
            env = ""
        raise OSError(
            "Precompiled binaries are not available for the current platform. "
            "Please reinstall from source using:\n\n"
            "  pip uninstall ctransformers --yes\n"
            f"  {env}pip install ctransformers --no-binary ctransformers\n\n"
        )
    return str(path)


def load_cuda() -> bool:
    try:
        import nvidia
    except ImportError:
        return False
    path = Path(nvidia.__path__[0])
    system = platform.system()
    if system == "Windows":
        libs = [
            path / "cuda_runtime" / "bin" / "cudart64_12.dll",
            path / "cublas" / "bin" / "cublas64_12.dll",
        ]
    else:
        libs = [
            path / "cuda_runtime" / "lib" / "libcudart.so.12",
            path / "cublas" / "lib" / "libcublas.so.12",
        ]
    for lib in libs:
        if not lib.is_file():
            return False
    libs = [str(lib.resolve()) for lib in libs]
    for lib in libs:
        CDLL(lib)
    return True
