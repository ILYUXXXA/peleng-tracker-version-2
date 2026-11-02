from peleng_tracker.core.vision_websocket import result_payload
from peleng_tracker.domain import Track, VisionResult, WorldTrack


def test_websocket_schema_v2_keeps_compatibility_alias() -> None:
    track = WorldTrack("g1", 0, "person", 0.8, [0, 1])
    local_track = Track("7", 0, (1, 2, 3, 4), 0.8, 0, "person", global_id="g1", is_target=True)
    payload = result_payload(VisionResult(1, 1.0, 2.0, [0], {0: []}, {0: [local_track]}, [track]))
    assert payload["schema_version"] == 2
    assert payload["world_tracks"] == payload["bev_tracks"]
    assert payload["all_streams"]["0"]["tracks"][0]["global_id"] == "g1"
    assert payload["all_streams"]["0"]["tracks"][0]["is_target"] is True
    assert payload["world_tracks"][0]["source_track_keys"] == []
    assert payload["association_diagnostics"] == {
        "local_tracks_by_camera": {},
        "fused_world_tracks": 0,
        "single_camera_fallback_tracks": 0,
    }
