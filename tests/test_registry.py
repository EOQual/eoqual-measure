"""Cohérence du registre des mesures."""
import importlib

import pytest

from eoqual.measure import MEASURES_CONFIGS, list_measures

REQUIRED_KEYS = {"family", "status", "input", "outputs", "entry", "builds_on", "reference"}


@pytest.mark.parametrize("name", list(MEASURES_CONFIGS))
def test_entry_is_complete(name):
    cfg = MEASURES_CONFIGS[name]
    assert REQUIRED_KEYS <= set(cfg)
    assert cfg["status"] in {"available", "planned"}
    assert set(cfg["builds_on"]) <= set(MEASURES_CONFIGS)


@pytest.mark.parametrize("name", [n for n, c in MEASURES_CONFIGS.items() if c["status"] == "available"])
def test_available_entry_point_resolves(name):
    module, func = MEASURES_CONFIGS[name]["entry"].split(":")
    assert callable(getattr(importlib.import_module(module), func))


def test_list_measures_prints(capsys):
    list_measures(include_planned=True)
    assert "mtf_slanted_edge" in capsys.readouterr().out
