"""Unit tests for the vehicle lookup (the quote agent passes whatever the user typed)."""
import pytest

from app.services.vehicle_catalog import lookup_vehicle


@pytest.mark.parametrize("query, expected_model", [
    ("swift", "Swift"),
    ("SWIFT", "Swift"),
    ("cret", "Creta"),
    ("Maruti Swift", "Swift"),
    ("Hyundai Creta", "Creta"),
    ("Swift Dzire", "Swift"),
    ("Toyota Innova", "Innova Crysta"),
    ("Mahindra XUV 700", "XUV700"),
    ("Maruti Wagon-R", "WagonR"),
])
def test_lookup_finds_model_with_or_without_brand(query, expected_model):
    """Brand names, spacing and case must not stop a catalog car from being found."""
    assert lookup_vehicle(query)["model"] == expected_model


def test_lookup_returns_none_for_unknown_car():
    """Cars outside the catalog return None so the agent can list supported models."""
    assert lookup_vehicle("BMW X5") is None
