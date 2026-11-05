"""AnonSpec strength conversion (no models needed)."""

from privacy_pipeline.attacks.reid_eval import AnonSpec


def test_absolute_strength_is_pixels():
    assert AnonSpec("pixelate", 12).pixel_strength(100) == 12
    assert AnonSpec("pixelate", 12).pixel_strength(30) == 12


def test_relative_pixelate_is_blocks_across_face():
    s = AnonSpec("pixelate", 8, "relative")
    assert s.pixel_strength(96) == 12
    assert s.pixel_strength(32) == 4


def test_relative_gaussian_is_percent_of_face_width():
    s = AnonSpec("gaussian", 50, "relative")
    assert s.pixel_strength(100) == 50
    assert s.pixel_strength(4) == 3  # floor of 3 px


def test_key_is_unique_per_setting():
    keys = {
        AnonSpec("pixelate", 8).key,
        AnonSpec("pixelate", 8, "relative").key,
        AnonSpec("pixelate", 8, padding=0).key,
    }
    assert len(keys) == 3
