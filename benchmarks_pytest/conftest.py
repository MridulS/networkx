"""Configuration for pytest-codspeed benchmarks."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark for CodSpeed"
    )
