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

@patch("utils.embed.DeepFace.represent")
def test_known_user(mock_represent,create_fake_db,monkeypatch):

    db_path,embedding = create_fake_db
    mock_represent.return_value = [{"embedding":embedding}]
    monkeypatch.chdir(db_path.parent)

    result = recognition(test_image)
    assert result==test_user


@patch("utils.embed.DeepFace.represent")
def test_unknown_user(mock_represent,create_fake_db,monkeypatch):
    db_path,embedding = create_fake_db
    mock_represent.return_value = [{"embedding":[0.0]*512}]
    monkeypatch.chdir(db_path.parent)
    result = recognition(test_image)
    assert result=="unknown"
    
