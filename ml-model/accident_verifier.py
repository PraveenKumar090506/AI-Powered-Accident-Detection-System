"""Verify possible accidents from tracked vehicles across consecutive frames.

This module scores pairwise vehicle interactions produced by
``vehicle_tracker.TrackedVehicle``. A single suspicious frame is never
treated as an accident; confirmation requires several frames in a row.

It does not send alerts or decide emergency response.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Protocol, Sequence


class VehicleMovementLike(Protocol):
    """Minimal movement fields used from ``TrackedVehicle.movement``."""

    dx: float
    dy: float
    distance: float


class TrackedVehicleLike(Protocol):
    """Minimal tracked-vehicle fields used by the verifier."""

    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    movement: VehicleMovementLike


@dataclass(frozen=True)
class AccidentVerificationResult:
    """Structured outcome for one analyzed frame."""

    accident_suspected: bool
    confidence: float
    involved_track_ids: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PairAssessment:
    """Per-pair score for the current frame. Easy to inspect or tweak later."""

    track_ids: tuple[int, int]
    score: float
    suspicious: bool
    reasons: tuple[str, ...]
    iou: float
    center_distance: float


def bbox_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_distance(
    center_a: tuple[float, float],
    center_b: tuple[float, float],
) -> float:
    """Euclidean distance between two vehicle centers."""
    return hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


class AccidentVerifier:
    """Detect possible accidents from tracked vehicles over time.

    Call :meth:`update` once per frame with the list returned by
    ``VehicleTracker.update``. Call :meth:`reset` before a new video.
    """

    def __init__(
        self,
        iou_threshold: float = 0.15,
        close_distance_px: float = 50.0,
        sudden_movement_px: float = 25.0,
        sudden_movement_delta_px: float = 20.0,
        pair_score_threshold: float = 0.55,
        min_consecutive_frames: int = 3,
        overlap_weight: float = 0.45,
        closeness_weight: float = 0.35,
        motion_weight: float = 0.20,
    ) -> None:
        """Create a verifier with tunable geometric and temporal thresholds.

        Args:
            iou_threshold: IoU at which boxes are treated as overlapping.
            close_distance_px: Center distance (pixels) treated as very close.
            sudden_movement_px: Movement distance in one frame treated as abrupt.
            sudden_movement_delta_px: Change vs the previous frame's movement.
            pair_score_threshold: Minimum combined score for a suspicious pair.
            min_consecutive_frames: Suspicious frames required before suspicion.
            overlap_weight: Contribution of IoU to the pair score.
            closeness_weight: Contribution of center proximity to the pair score.
            motion_weight: Contribution of sudden movement to the pair score.
        """
        if min_consecutive_frames < 2:
            raise ValueError("min_consecutive_frames must be at least 2")

        self.iou_threshold = iou_threshold
        self.close_distance_px = close_distance_px
        self.sudden_movement_px = sudden_movement_px
        self.sudden_movement_delta_px = sudden_movement_delta_px
        self.pair_score_threshold = pair_score_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.overlap_weight = overlap_weight
        self.closeness_weight = closeness_weight
        self.motion_weight = motion_weight

        # Consecutive suspicious-frame counts, keyed by sorted track-id pair.
        self._pair_streaks: dict[tuple[int, int], int] = {}
        # Previous per-track movement distance, used to detect sudden change.
        self._previous_movement: dict[int, float] = {}

    def reset(self) -> None:
        """Clear temporal state before analyzing a new video."""
        self._pair_streaks.clear()
        self._previous_movement.clear()

    def update(
        self,
        tracked_vehicles: Sequence[TrackedVehicleLike],
    ) -> AccidentVerificationResult:
        """Analyze one frame of tracked vehicles.

        A pair must stay suspicious for ``min_consecutive_frames`` before
        ``accident_suspected`` becomes True.
        """
        assessments = self._assess_pairs(tracked_vehicles)
        suspicious_pairs = {
            assessment.track_ids for assessment in assessments if assessment.suspicious
        }

        self._update_streaks(suspicious_pairs)
        confirmed = self._confirmed_pairs()

        self._previous_movement = {
            vehicle.track_id: float(vehicle.movement.distance)
            for vehicle in tracked_vehicles
        }

        if not confirmed:
            return AccidentVerificationResult(
                accident_suspected=False,
                confidence=0.0,
                involved_track_ids=(),
                reasons=(),
            )

        involved = sorted({track_id for pair in confirmed for track_id in pair})
        reasons = self._collect_reasons(assessments, confirmed)
        confidence = self._confidence(confirmed, assessments)
        return AccidentVerificationResult(
            accident_suspected=True,
            confidence=confidence,
            involved_track_ids=tuple(involved),
            reasons=tuple(reasons),
        )

    def _assess_pairs(
        self,
        tracked_vehicles: Sequence[TrackedVehicleLike],
    ) -> list[PairAssessment]:
        """Score every unique vehicle pair in the current frame."""
        assessments: list[PairAssessment] = []
        for vehicle_a, vehicle_b in combinations(tracked_vehicles, 2):
            assessments.append(self._assess_pair(vehicle_a, vehicle_b))
        return assessments

    def _assess_pair(
        self,
        vehicle_a: TrackedVehicleLike,
        vehicle_b: TrackedVehicleLike,
    ) -> PairAssessment:
        """Combine overlap, closeness, and motion into one pair score."""
        track_ids = self._pair_key(vehicle_a.track_id, vehicle_b.track_id)
        iou = bbox_iou(vehicle_a.bbox, vehicle_b.bbox)
        distance = center_distance(vehicle_a.center, vehicle_b.center)
        sudden_a = self._is_sudden_movement(vehicle_a)
        sudden_b = self._is_sudden_movement(vehicle_b)

        overlap_score = min(1.0, iou / self.iou_threshold) if self.iou_threshold else 0.0
        closeness_score = self._closeness_score(distance)
        motion_score = 1.0 if (sudden_a or sudden_b) else 0.0

        score = (
            self.overlap_weight * overlap_score
            + self.closeness_weight * closeness_score
            + self.motion_weight * motion_score
        )

        reasons: list[str] = []
        if iou >= self.iou_threshold:
            reasons.append(
                f"tracks {track_ids[0]} and {track_ids[1]} overlap (IoU={iou:.2f})"
            )
        if distance <= self.close_distance_px:
            reasons.append(
                f"tracks {track_ids[0]} and {track_ids[1]} are very close "
                f"({distance:.1f}px)"
            )
        if sudden_a:
            reasons.append(f"track {vehicle_a.track_id} had a sudden movement change")
        if sudden_b:
            reasons.append(f"track {vehicle_b.track_id} had a sudden movement change")

        # Geometric contact (overlap or very close) is required; motion boosts score.
        geometrically_involved = (
            iou >= self.iou_threshold or distance <= self.close_distance_px
        )
        suspicious = geometrically_involved and score >= self.pair_score_threshold
        return PairAssessment(
            track_ids=track_ids,
            score=score,
            suspicious=suspicious,
            reasons=tuple(reasons),
            iou=iou,
            center_distance=distance,
        )

    def _closeness_score(self, distance: float) -> float:
        """Map center distance to 0..1, peaking when vehicles nearly coincide."""
        if self.close_distance_px <= 0:
            return 0.0
        if distance >= self.close_distance_px:
            return 0.0
        return 1.0 - (distance / self.close_distance_px)

    def _is_sudden_movement(self, vehicle: TrackedVehicleLike) -> bool:
        """True if this track jumped or changed speed sharply versus last frame."""
        current = float(vehicle.movement.distance)
        previous = self._previous_movement.get(vehicle.track_id)
        if previous is None:
            return False
        if current >= self.sudden_movement_px:
            return True
        return abs(current - previous) >= self.sudden_movement_delta_px

    def _update_streaks(self, suspicious_pairs: set[tuple[int, int]]) -> None:
        """Increment streaks for active pairs; reset pairs that are no longer suspicious."""
        for pair in list(self._pair_streaks):
            if pair not in suspicious_pairs:
                del self._pair_streaks[pair]

        for pair in suspicious_pairs:
            self._pair_streaks[pair] = self._pair_streaks.get(pair, 0) + 1

    def _confirmed_pairs(self) -> list[tuple[int, int]]:
        """Pairs whose suspicious streak meets the multi-frame requirement."""
        return [
            pair
            for pair, streak in self._pair_streaks.items()
            if streak >= self.min_consecutive_frames
        ]

    def _collect_reasons(
        self,
        assessments: Sequence[PairAssessment],
        confirmed: Sequence[tuple[int, int]],
    ) -> list[str]:
        """Keep reasons only for pairs that passed the consecutive-frame check."""
        confirmed_set = set(confirmed)
        reasons: list[str] = []
        seen: set[str] = set()
        for assessment in assessments:
            if assessment.track_ids not in confirmed_set:
                continue
            streak = self._pair_streaks[assessment.track_ids]
            streak_reason = (
                f"tracks {assessment.track_ids[0]} and {assessment.track_ids[1]} "
                f"remained suspicious for {streak} consecutive frames"
            )
            for reason in (*assessment.reasons, streak_reason):
                if reason not in seen:
                    seen.add(reason)
                    reasons.append(reason)
        return reasons

    def _confidence(
        self,
        confirmed: Sequence[tuple[int, int]],
        assessments: Sequence[PairAssessment],
    ) -> float:
        """Blend pair score with how long the streak has lasted past the minimum."""
        assessment_by_pair = {item.track_ids: item for item in assessments}
        confidences: list[float] = []
        for pair in confirmed:
            streak = self._pair_streaks[pair]
            extra = streak - self.min_consecutive_frames
            temporal = min(1.0, 0.6 + 0.1 * extra)
            pair_score = assessment_by_pair[pair].score if pair in assessment_by_pair else 0.0
            confidences.append(min(1.0, temporal * max(pair_score, 0.5)))
        return round(max(confidences), 3)

    @staticmethod
    def _pair_key(track_a: int, track_b: int) -> tuple[int, int]:
        """Stable unordered pair identity."""
        return (track_a, track_b) if track_a < track_b else (track_b, track_a)
