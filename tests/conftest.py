import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def sample_cache(tmp_path_factory):
    """Sample caches of the synthetic streams go to a temporary directory, not the user's cache."""
    os.environ["ARTLAPSE_CACHE"] = str(tmp_path_factory.mktemp("cache"))


def pytest_collection_modifyitems(config, items):
    """Tests marked gpu run only where CuPy sees a CUDA device."""
    from artlapse import accumulate_gpu

    if accumulate_gpu.available():
        return
    skip = pytest.mark.skip(reason="needs CuPy and a CUDA device (uv sync --extra gpu)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
