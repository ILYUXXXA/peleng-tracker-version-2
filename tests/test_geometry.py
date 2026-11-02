import numpy as np

from peleng_tracker.geometry import CalibrationDocument, CalibrationStore, CameraCalibrationData


def camera(translation: list[float]) -> CameraCalibrationData:
    return CameraCalibrationData(
        image_size=(1280, 720),
        camera_matrix=[[900, 0, 640], [0, 900, 360], [0, 0, 1]],
        distortion_coefficients=[0, 0, 0, 0, 0],
        rotation_world_to_camera=np.eye(3).tolist(),
        translation_world_to_camera=translation,
    )


def project(store: CalibrationStore, camera_id: int, point: np.ndarray) -> tuple[float, float]:
    pixel = store.cameras[camera_id].projection @ np.append(point, 1)
    return tuple(pixel[:2] / pixel[2])


def test_recovers_known_3d_point_and_rejects_negative_depth() -> None:
    store = CalibrationStore(CalibrationDocument(cameras={0: camera([0, 0, 0]), 1: camera([-1, 0, 0])}))
    point = np.array([0.4, 0.2, 5.0])
    result = store.triangulate({0: project(store, 0, point), 1: project(store, 1, point)}, 0.1)
    assert result is not None
    assert np.allclose(result.position_xyz_m, point, atol=1e-6)
    behind = np.array([0.4, 0.2, -5.0])
    assert store.triangulate({0: project(store, 0, behind), 1: project(store, 1, behind)}, 0.1) is None


def test_planar_homography_projects_the_person_ground_anchor() -> None:
    data = camera([0, 0, 0])
    data.planar_homography = [[0.1, 0, -10], [0, 0.1, -20], [0, 0, 1]]
    store = CalibrationStore(CalibrationDocument(cameras={0: data}))

    assert store.planar_position(0, (120.0, 240.0)) == (2.0, 4.0, 0.0)
