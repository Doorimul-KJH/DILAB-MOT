"""Frame-level multi-object SORT tracker."""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab.core.types import Detection, TrackResult
from motlab.trackers.sort.association import associate_by_iou
from motlab.trackers.sort.iou import compute_iou_matrix
from motlab.trackers.sort.track import SortTrack


@dataclass
class SortTracker:
    """Manage multiple SORT tracks for one frame at a time."""

    max_age: int = 1
    min_hits: int = 3
    iou_threshold: float = 0.3
    tracks: list[SortTrack] = field(default_factory=list)
    next_track_id: int = 1
    frame_count: int = 0

    def __post_init__(self) -> None:
        if self.max_age < 1:
            raise ValueError("SortTracker max_age must be greater than or equal to 1.")
        if self.min_hits < 1:
            raise ValueError("SortTracker min_hits must be greater than or equal to 1.")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("SortTracker iou_threshold must be between 0.0 and 1.0.")

    def update(self, detections: list[Detection]) -> list[TrackResult]:
        """Update tracks from detections belonging to one frame."""
        detection_frame = self._validate_detection_frames(detections)
        self.frame_count += 1

        predicted_boxes = [track.predict() for track in self.tracks]
        if not detections:
            self._remove_expired_tracks()
            return []

        detection_boxes = [detection.bbox for detection in detections]
        iou_matrix = compute_iou_matrix(predicted_boxes, detection_boxes)
        association = associate_by_iou(iou_matrix, iou_threshold=self.iou_threshold)

        current_confidence_by_track_id: dict[int, float] = {}
        for track_index, detection_index in association.matches:
            track = self.tracks[track_index]
            detection = detections[detection_index]
            track.update(detection.bbox)
            current_confidence_by_track_id[track.track_id] = detection.confidence

        for detection_index in association.unmatched_detections:
            detection = detections[detection_index]
            track = SortTrack(track_id=self.next_track_id, initial_bbox=detection.bbox)
            self.next_track_id += 1
            self.tracks.append(track)
            current_confidence_by_track_id[track.track_id] = detection.confidence

        self._remove_expired_tracks()
        return self._build_results(
            frame=detection_frame,
            confidence_by_track_id=current_confidence_by_track_id,
        )

    def _validate_detection_frames(self, detections: list[Detection]) -> int | None:
        if not detections:
            return None

        frame = detections[0].frame
        if any(detection.frame != frame for detection in detections):
            raise ValueError("All detections passed to SortTracker.update must belong to the same frame.")
        return frame

    def _remove_expired_tracks(self) -> None:
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

    def _build_results(
        self,
        frame: int | None,
        confidence_by_track_id: dict[int, float],
    ) -> list[TrackResult]:
        if frame is None:
            return []

        results: list[TrackResult] = []
        for track in sorted(self.tracks, key=lambda item: item.track_id):
            if track.time_since_update != 0:
                continue
            if track.hits < self.min_hits and self.frame_count > self.min_hits:
                continue

            confidence = confidence_by_track_id.get(track.track_id, 1.0)
            results.append(
                TrackResult(
                    frame=frame,
                    track_id=track.track_id,
                    bbox=track.get_state(),
                    confidence=confidence,
                )
            )
        return results
