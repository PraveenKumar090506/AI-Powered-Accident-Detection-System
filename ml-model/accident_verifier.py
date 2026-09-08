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
        """Create a verifier with tunable geometric, kinematic, and temporal thresholds.

        Args:
            iou_threshold: IoU at which boxes are treated as overlapping.
            close_distance_px: Minimum center distance (pixels) treated as close.
            sudden_movement_px: Movement distance in one frame treated as abrupt.
            sudden_movement_delta_px: Change vs previous frame's movement treated as abrupt.
            pair_score_threshold: Minimum combined score for a suspicious pair.
            min_consecutive_frames: Suspicious frames required before confirmation.
            overlap_weight: Contribution of IoU to the pair score.
            closeness_weight: Contribution of proximity to the pair score.
            motion_weight: Contribution of motion/stopping anomalies to the pair score.
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
        # Previous per-track movement (dx, dy, distance), used to detect sudden changes.
        self._previous_movement: dict[int, tuple[float, float, float]] = {}
        # Previous pairwise distance, used to detect rapid convergence.
        self._previous_distance: dict[tuple[int, int], float] = {}
        # Consecutive stopped frames per track ID.
        self._stopped_frames: dict[int, int] = {}
        # Maximum speed seen per track ID, ensuring a vehicle was moving before stopping.
        self._max_speed_seen: dict[int, float] = {}
        # Memory of recent kinematic anomalies per pair, decaying over time.
        self._recent_motion_event: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        """Clear temporal state before analyzing a new video."""
        self._pair_streaks.clear()
        self._previous_movement.clear()
        self._previous_distance.clear()
        self._stopped_frames.clear()
        self._max_speed_seen.clear()
        self._recent_motion_event.clear()

    def update(
        self,
        tracked_vehicles: Sequence[TrackedVehicleLike],
    ) -> AccidentVerificationResult:
        """Analyze one frame of tracked vehicles.

        A pair must stay suspicious for ``min_consecutive_frames`` before
        ``accident_suspected`` becomes True.
        """
        self._update_vehicle_state(tracked_vehicles)

        assessments = self._assess_pairs(tracked_vehicles)
        suspicious_pairs = {
            assessment.track_ids for assessment in assessments if assessment.suspicious
        }

        self._update_streaks(suspicious_pairs)
        confirmed = self._confirmed_pairs()

        self._previous_movement = {
            vehicle.track_id: (
                float(vehicle.movement.dx),
                float(vehicle.movement.dy),
                float(vehicle.movement.distance),
            )
            for vehicle in tracked_vehicles
        }

        new_prev_dist: dict[tuple[int, int], float] = {}
        for va, vb in combinations(tracked_vehicles, 2):
            pkey = self._pair_key(va.track_id, vb.track_id)
            new_prev_dist[pkey] = center_distance(va.center, vb.center)
        self._previous_distance = new_prev_dist

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

    def _update_vehicle_state(self, tracked_vehicles: Sequence[TrackedVehicleLike]) -> None:
        """Track speeds and consecutive stationary frames for active vehicles."""
        current_ids = {v.track_id for v in tracked_vehicles}
        for vehicle in tracked_vehicles:
            tid = vehicle.track_id
            spd = float(vehicle.movement.distance)
            self._max_speed_seen[tid] = max(self._max_speed_seen.get(tid, 0.0), spd)
            if spd < 1.2:
                self._stopped_frames[tid] = self._stopped_frames.get(tid, 0) + 1
            else:
                self._stopped_frames[tid] = 0

        for tid in list(self._stopped_frames):
            if tid not in current_ids:
                del self._stopped_frames[tid]

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
        """Combine proximity, kinematics, direction change, and abnormal stopping into one pair score."""
        track_ids = self._pair_key(vehicle_a.track_id, vehicle_b.track_id)
        iou = bbox_iou(vehicle_a.bbox, vehicle_b.bbox)
        dist = center_distance(vehicle_a.center, vehicle_b.center)

        box_a = vehicle_a.bbox
        box_b = vehicle_b.bbox
        w_a = max(1, box_a[2] - box_a[0])
        h_a = max(1, box_a[3] - box_a[1])
        w_b = max(1, box_b[2] - box_b[0])
        h_b = max(1, box_b[3] - box_b[1])
        diag_a = hypot(w_a, h_a)
        diag_b = hypot(w_b, h_b)
        scale = max(1.0, (diag_a + diag_b) / 2.0)

        # Adaptive closeness based on bounding-box scale, capped to avoid screen-wide false matches.
        adaptive_close_px = min(220.0, max(self.close_distance_px, scale * 1.8))

        gap_x = max(0, max(box_a[0] - box_b[2], box_b[0] - box_a[2]))
        gap_y = max(0, max(box_a[1] - box_b[3], box_b[1] - box_a[3]))
        edge_dist = hypot(gap_x, gap_y)

        # 1. Spatial Proximity: box overlap, adaptive center distance, or edge-to-edge separation.
        overlap_score = min(1.0, iou / self.iou_threshold) if self.iou_threshold else 0.0
        center_closeness = max(0.0, 1.0 - (dist / adaptive_close_px)) if adaptive_close_px > 0 else 0.0
        edge_closeness = max(0.0, 1.0 - (edge_dist / (scale * 1.5)))
        closeness_score = max(center_closeness, edge_closeness)
        spatial_score = max(overlap_score, closeness_score)

        geometrically_involved = (
            dist <= 250.0
            and (
                iou >= self.iou_threshold
                or dist <= adaptive_close_px
                or edge_dist <= scale * 1.2
            )
        )

        reasons: list[str] = []
        if iou >= self.iou_threshold:
            reasons.append(f"tracks {track_ids[0]} and {track_ids[1]} overlap (IoU={iou:.2f})")
        if dist <= self.close_distance_px:
            reasons.append(f"tracks {track_ids[0]} and {track_ids[1]} are very close ({dist:.1f}px)")
        elif geometrically_involved:
            reasons.append(
                f"tracks {track_ids[0]} and {track_ids[1]} are in proximity "
                f"({dist:.1f}px, edge {edge_dist:.1f}px)"
            )

        # 2. Kinematic and Behavioral Anomaly Signals.
        sudden_a = self._is_sudden_movement(vehicle_a)
        sudden_b = self._is_sudden_movement(vehicle_b)
        dir_a = self._is_sudden_direction(vehicle_a)
        dir_b = self._is_sudden_direction(vehicle_b)

        prev_d = self._previous_distance.get(track_ids)
        rapid_approach = False
        if prev_d is not None and (prev_d - dist) >= max(5.0, 0.06 * scale):
            rapid_approach = True

        stop_a = self._stopped_frames.get(vehicle_a.track_id, 0)
        stop_b = self._stopped_frames.get(vehicle_b.track_id, 0)
        had_motion_a = self._max_speed_seen.get(vehicle_a.track_id, 0.0) >= 3.0
        had_motion_b = self._max_speed_seen.get(vehicle_b.track_id, 0.0) >= 3.0

        abnormal_stop = (
            geometrically_involved
            and (
                (stop_a >= 3 and had_motion_a and stop_b >= 3 and had_motion_b)
                or ((stop_a >= 8 and had_motion_a) or (stop_b >= 8 and had_motion_b))
            )
        )

        if sudden_a:
            reasons.append(f"track {vehicle_a.track_id} had a sudden movement change")
        if sudden_b:
            reasons.append(f"track {vehicle_b.track_id} had a sudden movement change")
        if dir_a:
            reasons.append(f"track {vehicle_a.track_id} had a sudden direction change")
        if dir_b:
            reasons.append(f"track {vehicle_b.track_id} had a sudden direction change")
        if rapid_approach:
            reasons.append(f"tracks {track_ids[0]} and {track_ids[1]} rapidly converged")
        if abnormal_stop:
            reasons.append(f"tracks {track_ids[0]} and {track_ids[1]} exhibited abnormal stopping")

        has_current_anomaly = (
            sudden_a or sudden_b or dir_a or dir_b or rapid_approach or abnormal_stop
        )

        if has_current_anomaly:
            self._recent_motion_event[track_ids] = 0
        elif track_ids in self._recent_motion_event:
            self._recent_motion_event[track_ids] += 1
            if self._recent_motion_event[track_ids] > 20:
                del self._recent_motion_event[track_ids]

        # Motion score bridges impact spikes with post-collision rest.
        motion_score = 0.0
        if has_current_anomaly:
            motion_score = 1.0
        elif track_ids in self._recent_motion_event:
            recency = 1.0 - (self._recent_motion_event[track_ids] / 20.0)
            motion_score = max(0.0, 0.8 * recency)

        # Composite score: requires both geometric proximity and kinematic anomaly.
        score = 0.45 * spatial_score + 0.55 * motion_score

        suspicious = (
            geometrically_involved
            and (motion_score >= 0.5)
            and (score >= self.pair_score_threshold)
        )

        return PairAssessment(
            track_ids=track_ids,
            score=round(score, 3),
            suspicious=suspicious,
            reasons=tuple(reasons),
            iou=iou,
            center_distance=dist,
        )

    def _closeness_score(self, distance: float) -> float:
        """Map center distance to 0..1, peaking when vehicles nearly coincide."""
        if self.close_distance_px <= 0:
            return 0.0
        if distance >= self.close_distance_px:
            return 0.0
        return 1.0 - (distance / self.close_distance_px)

    def _is_sudden_movement(self, vehicle: TrackedVehicleLike) -> bool:
        """True if this track jumped, braked, or changed speed sharply versus last frame."""
        current = float(vehicle.movement.distance)
        prev = self._previous_movement.get(vehicle.track_id)
        if prev is None:
            return current >= self.sudden_movement_px

        prev_spd = prev[2]
        delta = abs(current - prev_spd)
        box = vehicle.bbox
        diag = hypot(max(1, box[2] - box[0]), max(1, box[3] - box[1]))

        return (
            current >= self.sudden_movement_px
            or delta >= self.sudden_movement_delta_px
            or delta >= max(5.0, 0.06 * diag)
        )

    def _is_sudden_direction(self, vehicle: TrackedVehicleLike) -> bool:
        """True if this track swerved sharply while moving at significant speed."""
        current_spd = float(vehicle.movement.distance)
        prev = self._previous_movement.get(vehicle.track_id)
        if prev is None:
            return False

        prev_dx, prev_dy, prev_spd = prev
        if current_spd >= 4.0 and prev_spd >= 4.0:
            dot = vehicle.movement.dx * prev_dx + vehicle.movement.dy * prev_dy
            cos_theta = dot / (current_spd * prev_spd)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            if cos_theta < 0.2:  # > ~78 degrees sharp turn
                return True
        return False

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
