# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import pytest

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")

if int(zarr.__version__.split(".")[0]) < 3:
    pytest.skip("Icechunk needs zarr>=3", allow_module_level=True)

from anemoi.datasets.usage.store import dataset_lookup  # noqa: E402
from anemoi.datasets.usage.store import name_to_zarr_store  # noqa: E402
from anemoi.datasets.usage.store import open_zarr  # noqa: E402


@pytest.fixture
def repo_path(tmp_path):
    path = str(tmp_path / "repo")
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(path))
    session = repo.writable_session("main")
    group = zarr.open_group(session.store, mode="w", zarr_format=3)
    group.create_array("data", data=np.arange(12, dtype="float32").reshape(3, 4))
    group.attrs["version"] = 1
    first = session.commit("v1")
    repo.create_tag("v1", first)

    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="r+").attrs["version"] = 2
    session.commit("v2")
    return path, first


def test_branch_default(repo_path):
    path, _ = repo_path
    group = open_zarr(f"icechunk://{path}")
    assert group.attrs["version"] == 2
    assert np.array_equal(group["data"][:], np.arange(12, dtype="float32").reshape(3, 4))


def test_tag_and_snapshot_pin_a_version(repo_path):
    path, first = repo_path
    assert open_zarr(f"icechunk://{path}?tag=v1").attrs["version"] == 1
    assert open_zarr(f"icechunk://{path}?snapshot={first}").attrs["version"] == 1


def test_dataset_lookup_keeps_the_url(repo_path):
    path, _ = repo_path
    url = f"icechunk://{path}?tag=v1"
    assert dataset_lookup(url) == url


def test_store_is_read_only(repo_path):
    path, _ = repo_path
    with pytest.raises(Exception):
        zarr.open_group(name_to_zarr_store(f"icechunk://{path}"), mode="r+").attrs["x"] = 1


@pytest.mark.parametrize("suffix", ["?branch=main&tag=v1", "?bogus=1"])
def test_bad_selectors(repo_path, suffix):
    path, _ = repo_path
    with pytest.raises(ValueError):
        name_to_zarr_store(f"icechunk://{path}{suffix}")


def test_remote_hosts_not_supported():
    with pytest.raises(ValueError):
        name_to_zarr_store("icechunk://bucket/prefix")
