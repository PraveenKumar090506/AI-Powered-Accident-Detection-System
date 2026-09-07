"""Synthetic tests for AccidentVerifier using TrackedVehicle objects."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from accident_verifier import AccidentVerifier
from vehicle_tracker import TrackedVehicle, VehicleMovement

# Keep the consecutive-frame requirement explicit in every scenario.
MIN_CONSECUTIVE_FRAMES = 3


def make_vehicle(
    track_id: int,
    bbox: tuple[int, int, int, int],
    movement_distance: float = 0.0,
) -> TrackedVehicle:
    """Build a TrackedVehicle whose center matches the given box."""
    x1, y1, x2, y2 = bbox
    return TrackedVehicle(
        track_id=track_id,
        class_name="car",
        confidence=0.9,
        bbox=bbox,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        movement=VehicleMovement(dx=0.0, dy=0.0, distance=movement_distance),
    )


def far_apart_pair() -> list[TrackedVehicle]:
    """Two cars well beyond the default closeness distance."""
    return [
        make_vehicle(1, (0, 0, 40, 40)),
        make_vehicle(2, (400, 0, 440, 40)),
    ]


def overlapping_pair(movement_distance: float = 30.0) -> list[TrackedVehicle]:
    """Two cars with overlapping boxes and nearby centers."""
    return [
        make_vehicle(1, (100, 100, 180, 180), movement_distance=movement_distance),
        make_vehicle(2, (120, 120, 200, 200), movement_distance=movement_distance),
    ]


def new_verifier() -> AccidentVerifier:
    return AccidentVerifier(min_consecutive_frames=MIN_CONSECUTIVE_FRAMES)


def check(name: str, condition: bool, detail: str = "") -> bool:
    suffix = f" — {detail}" if detail else ""
    if condition:
        print(f"PASS: {name}{suffix}")
        return True
    print(f"FAIL: {name}{suffix}")
    return False


def test_far_apart_is_not_an_accident() -> bool:
    verifier = new_verifier()
    result = verifier.update(far_apart_pair())
    return check(
        "two vehicles far apart → no accident",
        result.accident_suspected is False,
        f"accident_suspected={result.accident_suspected}",
    )


def test_single_suspicious_frame_is_not_an_accident() -> bool:
    verifier = new_verifier()
    result = verifier.update(overlapping_pair())
    return check(
        "two vehicles overlapping for only 1 frame → no accident",
        result.accident_suspected is False,
        f"accident_suspected={result.accident_suspected}",
    )


def test_fewer_than_required_frames_is_not_an_accident() -> bool:
    verifier = new_verifier()
    results = [
        verifier.update(overlapping_pair())
        for _ in range(MIN_CONSECUTIVE_FRAMES - 1)
    ]
    any_suspected = any(result.accident_suspected for result in results)
    return check(
        "overlapping for fewer than required consecutive frames → no accident",
        any_suspected is False,
        f"frames={MIN_CONSECUTIVE_FRAMES - 1}, any_suspected={any_suspected}",
    )


def test_required_consecutive_frames_suspects_accident() -> bool:
    verifier = new_verifier()
    last_result = None
    for _ in range(MIN_CONSECUTIVE_FRAMES):
        last_result = verifier.update(overlapping_pair())

    asserted = (
        last_result is not None
        and last_result.accident_suspected is True
        and set(last_result.involved_track_ids) == {1, 2}
    )
    detail = (
        "no result"
        if last_result is None
        else (
            f"accident_suspected={last_result.accident_suspected}, "
            f"involved={last_result.involved_track_ids}"
        )
    )
    return check(
        "overlapping for required consecutive frames → accident suspected",
        asserted,
        detail,
    )


def test_normal_frame_resets_suspicion_streak() -> bool:
    verifier = new_verifier()

    # Build a streak just below the confirmation threshold.
    for _ in range(MIN_CONSECUTIVE_FRAMES - 1):
        early = verifier.update(overlapping_pair())
        if early.accident_suspected:
            return check(
                "a pair becomes normal again → suspicion streak resets",
                False,
                "accident was already suspected before the reset frame",
            )

    # One normal (far-apart) frame should clear the pair streak.
    reset_result = verifier.update(far_apart_pair())
    if reset_result.accident_suspected:
        return check(
            "a pair becomes normal again → suspicion streak resets",
            False,
            "accident still suspected on the normal frame",
        )

    # The same pair overlapping again must start from streak 1, not resume.
    after_reset = [
        verifier.update(overlapping_pair())
        for _ in range(MIN_CONSECUTIVE_FRAMES - 1)
    ]
    resumed_too_early = any(result.accident_suspected for result in after_reset)
    return check(
        "a pair becomes normal again → suspicion streak resets",
        resumed_too_early is False,
        f"suspected too early after reset={resumed_too_early}",
    )


def main() -> int:
    print("Accident verifier synthetic tests")
    print(f"Required consecutive frames: {MIN_CONSECUTIVE_FRAMES}")
    print()

    outcomes = [
        test_far_apart_is_not_an_accident(),
        test_single_suspicious_frame_is_not_an_accident(),
        test_fewer_than_required_frames_is_not_an_accident(),
        test_required_consecutive_frames_suspects_accident(),
        test_normal_frame_resets_suspicion_streak(),
    ]

    passed = sum(outcomes)
    total = len(outcomes)
    print()
    print(f"Summary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
