import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--lib",
        action="store",
        choices=("avx2", "avx", "basic", "cuda", "none"),
    )


@pytest.fixture
def lib(request):
    lib = request.config.getoption("--lib")
    return lib if lib != "none" else None
