from unittest.mock import patch
import pytest
import json
import numpy as np
from utils.embed import recognition
from PIL import Image
test_user = "keyur"
test_image  = "left.jpg"
@pytest.fixture
def create_fake_db(tmp_path):
    db_path = tmp_path/"face_embeddings.json"
    embedding = [0.1]*512
    with open(db_path,"w") as f:
        json.dump({test_user:[embedding]},f)
    return db_path,embedding

@patch("utils.embed.Image.open")
@patch("utils.embed.DeepFace.represent")
def test_known_user(mock_represent,mock_open,create_fake_db,monkeypatch):

    db_path,embedding = create_fake_db
    fake_img = Image.fromarray(
        np.zeros((224, 224, 3), dtype=np.uint8)
    )

    mock_open.return_value = fake_img
    mock_represent.return_value = [{"embedding":embedding}]
    monkeypatch.chdir(db_path.parent)

    result = recognition(test_image)
    assert result==test_user

@patch("utils.embed.Image.open")
@patch("utils.embed.DeepFace.represent")
def test_unknown_user(mock_represent,mock_open,create_fake_db,monkeypatch):
    db_path,embedding = create_fake_db
    fake_img = Image.fromarray(
        np.zeros((224, 224, 3), dtype=np.uint8)
    )
    mock_open.return_value = fake_img
    mock_represent.return_value = [{"embedding":[0.0]*512}]
    monkeypatch.chdir(db_path.parent)
    result = recognition(test_image)
    assert result=="unknown"
    
