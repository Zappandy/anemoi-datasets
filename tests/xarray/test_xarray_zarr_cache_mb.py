# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the cache_mb option in the xarray-zarr source loader.

The cache_mb option wraps the underlying LocalStore with zarr's experimental
CacheStore backed by an in-memory MemoryStore.  This avoids redundant chunk
reads when a store's storage chunks span multiple logical dimensions:

  ARCO PL chunks: (1, 13, 721, 1440) = 54 MB
  Without cache : iterating 13 levels reads the same 54 MB chunk 13x from NFS.
  With cache    : first level fetches; levels 1-12 are served from RAM.

The tests here use a synthetic zarr store written with xarray so that the
metadata is correct, and verify that CacheStore hits accumulate as expected.
"""

import numpy as np
import pytest
import tempfile
import os
import zarr
import xarray as xr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore


def _make_multi_level_store(tmpdir, n_times=4, n_levels=13, ny=5, nx=5):
    """Create a synthetic xarray zarr store with multi-level chunks."""
    path = os.path.join(tmpdir, "test_pl.zarr")
    data = np.random.rand(n_times, n_levels, ny, nx).astype("f4")
    ds = xr.Dataset({
        "z": xr.DataArray(data, dims=["time", "level", "lat", "lon"]),
        "t": xr.DataArray(data * 2, dims=["time", "level", "lat", "lon"]),
    })
    # Write with chunks that span ALL levels (mimics ARCO PL storage)
    ds.to_zarr(path, mode="w", encoding={
        "z": {"chunks": [1, n_levels, ny, nx]},
        "t": {"chunks": [1, n_levels, ny, nx]},
    })
    return path, ds


def test_cache_store_hits_on_level_iteration():
    """CacheStore accumulates hits when iterating levels of a multi-level chunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _make_multi_level_store(tmpdir)

        raw = zarr.open_group(path, mode="r")
        cache_backend = MemoryStore()
        cached = CacheStore(raw.store, cache_store=cache_backend, max_size=256 * 1024 * 1024)

        ds = xr.open_dataset(cached, engine="zarr", consolidated=False)
        for lvl in range(13):
            ds["z"].isel(time=0, level=lvl).values

        stats = cached.cache_stats()
        assert stats["hits"] > 0, (
            f"Expected cache hits when iterating 13 levels of a single chunk; "
            f"got hits={stats['hits']} misses={stats['misses']}"
        )
        assert stats["hit_rate"] > 0.5, (
            f"Hit rate {stats['hit_rate']:.2f} too low; chunk caching may not be effective"
        )


def test_cache_store_values_correct():
    """Values read through CacheStore match values read from the raw store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, ref_ds = _make_multi_level_store(tmpdir, n_times=2, n_levels=3)

        raw = zarr.open_group(path, mode="r")
        cached = CacheStore(raw.store, cache_store=MemoryStore(), max_size=64 * 1024 * 1024)
        ds_cached = xr.open_dataset(cached, engine="zarr", consolidated=False)
        ds_raw = xr.open_dataset(path, engine="zarr", consolidated=False)

        for t in range(2):
            for lvl in range(3):
                np.testing.assert_array_equal(
                    ds_cached["z"].isel(time=t, level=lvl).values,
                    ds_raw["z"].isel(time=t, level=lvl).values,
                    err_msg=f"Mismatch at time={t} level={lvl}",
                )


def test_cache_store_respects_max_size():
    """Cache stays within max_size budget (entries evicted when full)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _make_multi_level_store(tmpdir, n_times=10, n_levels=13)

        raw = zarr.open_group(path, mode="r")
        # Very small cache — will force evictions
        tiny_cache = CacheStore(raw.store, cache_store=MemoryStore(), max_size=1024)
        ds = xr.open_dataset(tiny_cache, engine="zarr", consolidated=False)

        # Should not raise even though cache overflows
        for t in range(10):
            ds["z"].isel(time=t, level=0).values

        info = tiny_cache.cache_info()
        assert info["current_size"] <= 1024 + 512, (
            f"Cache exceeded max_size: {info['current_size']} bytes"
        )
