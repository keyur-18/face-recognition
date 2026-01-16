from utils.enroll_utils import image_validation,save_images,creat_user_dir
import os
import tempfile
from PIL import Image
import io

def test_image_validation():
    images = {"front":1,"left":2,"right":3}
    assert image_validation(images) is True

def test_create_user_dir():
    with tempfile.TemporaryDirectory()  as tmpdir:
        path = creat_user_dir(tmpdir,"keyur")
        assert os.path.exists(path)


def create_dummy_img():
    img = Image.new("RGB",(100,100),color="red")
    b = io.BytesIO()
    img.save(b,format="JPEG")
    b.seek(0)
    return b
def test_image():
    images = {
        "front": create_dummy_img(),
        "left": create_dummy_img(),
        "right": create_dummy_img(),
    }
    with tempfile.TemporaryDirectory() as tempdir:
        save_images(images=images,user_dir=tempdir)
        assert os.path.exists(os.path.join(tempdir,"front.jpg"))
        assert os.path.exists(os.path.join(tempdir,"right.jpg"))
        assert os.path.exists(os.path.join(tempdir,"left.jpg"))
