import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--lib",
        action="store",
        choices=("avx2", "avx", "basic", "cuda"),
        required=True,
    )


@pytest.fixture
def lib(request):
    return request.config.getoption("--lib")
