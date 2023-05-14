from typing import Optional
import platform
from pathlib import Path


def find_library(path: Optional[str] = None) -> str:
    lib_directory = Path(__file__).parent.resolve() / 'lib'

    if path:
        subdirs = [d.name for d in lib_directory.iterdir() if d.is_dir()]
        if path not in subdirs:
            return path

    system = platform.system()
    if not path:
        if platform.processor() == 'arm':
            # Apple silicon doesn't support AVX/AVX2.
            path = 'basic' if system == 'Darwin' else ''
        else:
            path = 'avx2'

    name = 'ctransformers'
    if system == 'Linux':
        name = f'lib{name}.so'
    elif system == 'Windows':
        name = f'{name}.dll'
    elif system == 'Darwin':
        name = f'lib{name}.dylib'
    else:
        name = ''

    path = lib_directory / path / name
    if not path.is_file():
        raise OSError('The current platform is not supported. ' +
                      'Please try building the C++ library from source.')
    return str(path)
