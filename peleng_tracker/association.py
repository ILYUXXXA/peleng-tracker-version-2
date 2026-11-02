"""Cross-camera association using global one-to-one geometry assignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import lap
import numpy as np

from .domain import AssociationDiagnostics, AssociationPair, Track, WorldTrack
from .geometry import CalibrationStore, TriangulationResult


@dataclass(slots=True)
class _Candidate:
    triangulation: TriangulationResult
    ground_distance_m: float | None
    cost: float


class _TrackGroups:
    def __init__(self, tracks: list[Track]) -> None:
        self.parent = {track.key: track.key for track in tracks}
        self.cameras = {track.key: {track.camera_id} for track in tracks}

    def find(self, key: tuple[int, str]) -> tuple[int, str]:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: tuple[int, str], right: tuple[int, str]) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return True
        if self.cameras[left_root] & self.cameras[right_root]:
            return False
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.cameras[left_root].update(self.cameras.pop(right_root))
        return True


class CrossCameraAssociator:
    def __init__(
        self,
        calibration: CalibrationStore,
        max_reprojection_error_px: float,
        max_ground_distance_m: float = 1.5,
    ) -> None:
        self.calibration = calibration
        self.max_reprojection_error_px = max_reprojection_error_px
        self.max_ground_distance_m = max_ground_distance_m
        self.key_to_global: dict[tuple[int, str], str] = {}
        self.last_diagnostics = AssociationDiagnostics()
        self._next_id = 1
        self._trajectory: dict[str, list[tuple[float, float, float, float]]] = {}

    def _new_id(self) -> str:
        value = str(self._next_id)
        self._next_id += 1
        return value

    def _candidate(self, left: Track, right: Track) -> _Candidate | None:
        left_position = self.calibration.planar_position(left.camera_id, left.geometry_anchor)
        right_position = self.calibration.planar_position(right.camera_id, right.geometry_anchor)
        ground_distance: float | None = None
        if left_position is not None and right_position is not None:
            ground_distance = float(np.linalg.norm(np.subtract(left_position[:2], right_position[:2])))
            if ground_distance > self.max_ground_distance_m:
                return None

        triangulation = self.calibration.triangulate(
            {left.camera_id: left.geometry_anchor, right.camera_id: right.geometry_anchor},
            self.max_reprojection_error_px,
        )
        if triangulation is None:
            return None

        distance_cost = ground_distance / self.max_ground_distance_m if ground_distance is not None else 0.0
        reprojection_cost = (
            triangulation.reprojection_error_px / self.max_reprojection_error_px
            if self.max_reprojection_error_px > 0
            else 0.0
        )
        return _Candidate(triangulation, ground_distance, distance_cost + reprojection_cost)

    def _assign_camera_pair(self, left_tracks: list[Track], right_tracks: list[Track]) -> list[AssociationPair]:
        if not left_tracks or not right_tracks:
            return []
        costs = np.full((len(left_tracks), len(right_tracks)), 1e6, dtype=np.float64)
        candidates: dict[tuple[int, int], _Candidate] = {}
        for left_index, left in enumerate(left_tracks):
            for right_index, right in enumerate(right_tracks):
                candidate = self._candidate(left, right)
                if candidate is not None:
                    costs[left_index, right_index] = candidate.cost
                    candidates[(left_index, right_index)] = candidate

        if not candidates:
            return []
        _, assignments, _ = lap.lapjv(costs, extend_cost=True, cost_limit=2.0 + 1e-9)
        pairs = []
        for left_index, right_index in enumerate(assignments):
            candidate = candidates.get((left_index, int(right_index)))
            if candidate is None:
                continue
            left, right = left_tracks[left_index], right_tracks[int(right_index)]
            pairs.append(
                AssociationPair(
                    left.camera_id,
                    left.local_track_id,
                    right.camera_id,
                    right.local_track_id,
                    candidate.ground_distance_m,
                    candidate.triangulation.reprojection_error_px,
                    candidate.cost,
                )
            )
        return pairs

    def _groups_for_class(self, tracks: list[Track]) -> tuple[list[list[Track]], list[AssociationPair]]:
        ordered = sorted(tracks, key=lambda track: (track.camera_id, track.local_track_id))
        tracks_by_camera: dict[int, list[Track]] = defaultdict(list)
        for track in ordered:
            tracks_by_camera[track.camera_id].append(track)

        groups = _TrackGroups(ordered)
        selected_pairs: list[AssociationPair] = []
        for left_camera, right_camera in combinations(sorted(tracks_by_camera), 2):
            pairs = self._assign_camera_pair(tracks_by_camera[left_camera], tracks_by_camera[right_camera])
            for pair in sorted(pairs, key=lambda item: (item.cost, item.left_local_track_id, item.right_local_track_id)):
                left_key = (pair.left_camera_id, pair.left_local_track_id)
                right_key = (pair.right_camera_id, pair.right_local_track_id)
                if groups.union(left_key, right_key):
                    selected_pairs.append(pair)

        grouped: dict[tuple[int, str], list[Track]] = defaultdict(list)
        for track in ordered:
            grouped[groups.find(track.key)].append(track)
        return list(grouped.values()), selected_pairs

    def associate(self, tracks_by_camera: dict[int, list[Track]], timestamp: float | None = None) -> list[WorldTrack]:
        by_class: dict[int, list[Track]] = defaultdict(list)
        for tracks in tracks_by_camera.values():
            for track in tracks:
                by_class[track.class_id].append(track)

        groups_and_pairs: list[tuple[list[Track], list[AssociationPair]]] = []
        for class_id in sorted(by_class):
            groups, pairs = self._groups_for_class(by_class[class_id])
            for group in groups:
                keys = {track.key for track in group}
                group_pairs = [
                    pair
                    for pair in pairs
                    if (pair.left_camera_id, pair.left_local_track_id) in keys
                    and (pair.right_camera_id, pair.right_local_track_id) in keys
                ]
                groups_and_pairs.append((group, group_pairs))

        world_tracks: list[WorldTrack] = []
        used_global_ids: set[str] = set()
        for group, association_pairs in groups_and_pairs:
            known_ids = sorted(
                {self.key_to_global[track.key] for track in group if track.key in self.key_to_global}, key=int
            )
            global_id = next((item for item in known_ids if item not in used_global_ids), None)
            if global_id is None:
                global_id = self._new_id()
            used_global_ids.add(global_id)
            for track in group:
                self.key_to_global[track.key] = global_id
                track.global_id = global_id

            observations = {track.camera_id: track.geometry_anchor for track in group}
            triangulated = self.calibration.triangulate(observations, self.max_reprojection_error_px)
            position = triangulated.position_xyz_m if triangulated else None
            status = "valid" if triangulated else "unavailable"
            reprojection_error = triangulated.reprojection_error_px if triangulated else None
            if position is None and len(group) == 1:
                position = self.calibration.planar_position(group[0].camera_id, group[0].geometry_anchor)
                if position is not None:
                    status = "planar_fallback"

            trajectory = self._trajectory.setdefault(global_id, [])
            velocity = (0.0, 0.0, 0.0)
            if position is not None and timestamp is not None:
                if trajectory and timestamp > trajectory[-1][3]:
                    elapsed = timestamp - trajectory[-1][3]
                    velocity = tuple((position[index] - trajectory[-1][index]) / elapsed for index in range(3))
                trajectory.append((*position, timestamp))
                del trajectory[:-100]

            world_tracks.append(
                WorldTrack(
                    global_id=global_id,
                    class_id=group[0].class_id,
                    class_name=group[0].class_name,
                    confidence=sum(track.confidence for track in group) / len(group),
                    source_cameras=sorted({track.camera_id for track in group}),
                    position_xyz_m=position,
                    velocity_xyz_mps=velocity,
                    reprojection_error_px=reprojection_error,
                    position_status=status,
                    trajectory=trajectory.copy(),
                    source_track_keys=[f"{track.camera_id}:{track.local_track_id}" for track in group],
                    association_pairs=association_pairs,
                )
            )

        self.last_diagnostics = AssociationDiagnostics(
            local_tracks_by_camera={camera_id: len(tracks) for camera_id, tracks in sorted(tracks_by_camera.items())},
            fused_world_tracks=sum(len(track.source_cameras) > 1 for track in world_tracks),
            single_camera_fallback_tracks=sum(
                len(track.source_cameras) == 1 and track.position_status == "planar_fallback" for track in world_tracks
            ),
        )
        return world_tracks

    def reset(self) -> None:
        self.key_to_global.clear()
        self.last_diagnostics = AssociationDiagnostics()
        self._next_id = 1
        self._trajectory.clear()
