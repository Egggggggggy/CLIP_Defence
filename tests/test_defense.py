from PIL import Image

from src.defense import preprocess_image


def test_preprocess_image_keeps_size_and_mode() -> None:
    img = Image.new("RGB", (32, 32), color=(120, 33, 64))
    out = preprocess_image(img, quality=70)
    assert out.size == (32, 32)
    assert out.mode == "RGB"
