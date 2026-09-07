"""PASS/FAIL tests for LocationProvider."""

from __future__ import annotations

import sys
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from location_service import LocationProvider


def check(name: str, condition: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    if condition:
        print(f"PASS: {name}{suffix}")
        return True
    print(f"FAIL: {name}{suffix}")
    return False


def main() -> int:
    print("LocationProvider tests")
    print()

    location = LocationProvider().get_location()

    outcomes = [
        check("returned value is a dictionary", isinstance(location, dict), type(location).__name__),
        check(
            "contains latitude and longitude",
            "latitude" in location and "longitude" in location,
            str(list(location.keys())),
        ),
        check("latitude is 13.0827", location.get("latitude") == 13.0827, str(location.get("latitude"))),
        check("longitude is 80.2707", location.get("longitude") == 80.2707, str(location.get("longitude"))),
    ]

    passed = sum(outcomes)
    total = len(outcomes)
    print()
    print(f"Summary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
