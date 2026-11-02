from pathlib import Path

import numpy as np

from peleng_tracker.epfl_calibration import convert_epfl_calibration
from peleng_tracker.geometry import CalibrationStore

EPFL_SAMPLE = """
# Camera 0
# Ground plane homography
5 0 0
0 5 0
0 0 1
# Head plane homography
6 0 0
0 6 0
0 0 1
# Camera 1
# Ground plane homography
5 0 100
0 5 0
0 0 1
# Head plane homography
6 0 100
0 6 0
0 0 1
"""


def test_epfl_conversion_preserves_ground_projection_and_metric_planar_position(tmp_path: Path) -> None:
    source = tmp_path / "calibration.txt"
    source.write_text(EPFL_SAMPLE, encoding="utf-8")
    document = convert_epfl_calibration(
        source,
        camera_ids=[0, 1],
        image_size=(360, 288),
        head_height_m=1.0,
        ground_units_per_meter=100.0,
    )
    store = CalibrationStore(document)

    assert store.status()["valid"] is True
    assert store.cameras[0].camera_forward_axis == "negative_z"
    assert np.allclose(store.planar_position(0, (40.0, 60.0)), (2.0, 3.0, 0.0))
    for camera_id in (0, 1):
        model = store.cameras[camera_id]
        point = np.array([2.0, 3.0, 0.0, 1.0])
        pixel = model.projection @ point
        pixel = pixel[:2] / pixel[2]
        expected_x = 40.0 if camera_id == 0 else 20.0
        assert np.allclose(pixel, (expected_x, 60.0))
