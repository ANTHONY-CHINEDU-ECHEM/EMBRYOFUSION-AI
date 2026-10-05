from __future__ import annotations

import pytest

from embryofusion.data.simulate import generate_paired_dataset


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired")
    generate_paired_dataset(root, 320, render_size=96, image_size=32, seed=9)
    return root
