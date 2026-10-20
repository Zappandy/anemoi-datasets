# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from unittest.mock import MagicMock

import earthkit.data as ekd
import pytest

from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.polytope import PolytopeSource

REQUEST = {
    "class": "d1",
    "dataset": "on-demand-extremes-dt",
    "expver": "0099",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "georef": "u4usq2",
    "param": [167],
}

ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"

_REAL_FROM_SOURCE = ekd.from_source


@pytest.fixture
def from_source(monkeypatch):
    """Replace ``ekd.from_source`` so that no network access happens."""
    mock = MagicMock(side_effect=lambda name, *args, **kwargs: _REAL_FROM_SOURCE("empty"))
    monkeypatch.setattr(ekd, "from_source", mock)
    return mock


def _polytope_calls(mock):
    return [c for c in mock.call_args_list if c.args[0] == "polytope"]


def test_polytope_source_is_registered():
    assert source_registry.lookup("polytope") is PolytopeSource


def test_polytope_requests_use_collection_address_and_stream(from_source):
    source = PolytopeSource(None, request=REQUEST, address=ADDRESS)
    dates = ValidDates([datetime.datetime(2025, 6, 16, 0, 0)])

    source.execute_valid_dates(dates)

    (call,) = _polytope_calls(from_source)
    assert call.args[1] == "destination-earth"
    assert call.kwargs == {"address": ADDRESS, "stream": False}
    assert call.args[2]["date"] == "20250616"
    assert call.args[2]["time"] == "0000"
    assert call.args[2]["step"] == 0
    assert call.args[2]["georef"] == "u4usq2"


def test_polytope_collection_can_be_overridden(from_source):
    source = PolytopeSource(None, request=REQUEST, collection="ecmwf-mars")

    source.execute_valid_dates(ValidDates([datetime.datetime(2025, 6, 16, 12, 0)]))

    (call,) = _polytope_calls(from_source)
    assert call.args[1] == "ecmwf-mars"
    assert "address" not in call.kwargs
    assert call.args[2]["time"] == "1200"


def test_polytope_step_is_not_overridden(from_source):
    source = PolytopeSource(None, request={**REQUEST, "step": 12})

    source.execute_valid_dates(ValidDates([datetime.datetime(2025, 6, 16, 0, 0)]))

    (call,) = _polytope_calls(from_source)
    assert call.args[2]["step"] == 12


def test_polytope_does_not_modify_the_input_request(from_source):
    request = dict(REQUEST)
    source = PolytopeSource(None, request=request)

    source.execute_valid_dates(ValidDates([datetime.datetime(2025, 6, 16, 0, 0)]))

    assert request == REQUEST


def test_polytope_several_dates_are_compressed(from_source):
    source = PolytopeSource(None, request=REQUEST)
    dates = ValidDates([datetime.datetime(2025, 6, 16, h) for h in (0, 6, 12, 18)])

    source.execute_valid_dates(dates)

    calls = _polytope_calls(from_source)
    assert len(calls) < len(dates)
    times = set()
    for call in calls:
        value = call.args[2]["time"]
        times.update(value if isinstance(value, (list, tuple)) else [value])
    assert times == {"0000", "0600", "1200", "1800"}


def test_polytope_retrieval_error_is_propagated(from_source):
    def fail(name, *args, **kwargs):
        if name == "polytope":
            raise RuntimeError("polytope unavailable")
        return _REAL_FROM_SOURCE("empty")

    from_source.side_effect = fail
    source = PolytopeSource(None, request=REQUEST)

    with pytest.raises(RuntimeError, match="polytope unavailable"):
        source.execute_valid_dates(ValidDates([datetime.datetime(2025, 6, 16, 0, 0)]))
