"""Mock location provider for accident events.

Returns a static location until a real GPS or camera-site source is wired in.
This module does not call external APIs.
"""

from __future__ import annotations

from typing import TypedDict

# Chennai (placeholder until camera geolocation is available).
MOCK_LATITUDE: float = 13.0827
MOCK_LONGITUDE: float = 80.2707


class Location(TypedDict):
    latitude: float
    longitude: float


class LocationProvider:
    """Supplies latitude and longitude for emergency event context."""

    def get_location(self) -> Location:
        """Return the current mock location as a simple dictionary."""
        return {
            "latitude": MOCK_LATITUDE,
            "longitude": MOCK_LONGITUDE,
        }
