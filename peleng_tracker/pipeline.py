"""Application service orchestrating detection, tracking and world estimation."""

from __future__ import annotations

import time

from .association import CrossCameraAssociator
from .domain import FramePacket, VisionResult
from .ports import LocalTracker, ObjectDetector
from .targets import TargetManager


class VisionPipeline:
    def __init__(
        self,
        detector: ObjectDetector,
        tracker: LocalTracker,
        associator: CrossCameraAssociator,
        targets: TargetManager,
        max_sync_skew_ms: float,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.associator = associator
        self.targets = targets
        self.max_sync_skew_ms = max_sync_skew_ms
        self.frame_id = 0

    def process(self, packets: list[FramePacket]) -> VisionResult | None:
        if not packets:
            return None
        timestamps = [packet.captured_at for packet in packets]
        skew_ms = (max(timestamps) - min(timestamps)) * 1000.0
        if skew_ms > self.max_sync_skew_ms:
            newest = max(timestamps)
            packets = [packet for packet in packets if (newest - packet.captured_at) * 1000.0 <= self.max_sync_skew_ms]
        if not packets:
            return None

        started = time.perf_counter()
        detections_by_camera = {}
        tracks_by_camera = {}
        for packet in packets:
            detections = self.detector.detect(packet.frame)
            detections_by_camera[packet.camera_id] = detections
            tracks_by_camera[packet.camera_id] = self.tracker.update(
                detections, packet.frame, packet.camera_id, packet.captured_at
            )

        world_tracks = self.associator.associate(tracks_by_camera, max(packet.captured_at for packet in packets))
        self.targets.apply(world_tracks, self.associator.key_to_global)
        target_global_ids = {track.global_id for track in world_tracks if track.is_target}
        for tracks in tracks_by_camera.values():
            for track in tracks:
                track.is_target = track.global_id in target_global_ids
        self.frame_id += 1
        return VisionResult(
            frame_id=self.frame_id,
            timestamp=max(packet.captured_at for packet in packets),
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            active_stream_ids=[packet.camera_id for packet in packets],
            detections=detections_by_camera,
            tracks=tracks_by_camera,
            world_tracks=world_tracks,
            sync_skew_ms=skew_ms,
            association_diagnostics=self.associator.last_diagnostics,
        )

    def reset(self) -> None:
        self.tracker.reset()
        self.associator.reset()
        self.frame_id = 0
