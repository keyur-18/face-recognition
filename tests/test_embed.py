import numpy as np
from utils.embed import identity

def test_identity_face_correct():
    embeddings = {'Dhruvil':[[0.1]*512]}
    test_emb  = [0.2]*512
    result  = identity(test_emb=test_emb,embeddings=embeddings)
    assert result == "Dhruvil"

def test_identity_face_unknown():
    embeddings = {'Dhruvil':[[0.0]*512]}
    test_emb  = [1.0]*512
    result  = identity(test_emb=test_emb,embeddings=embeddings)
    assert result == "unknown"

def test_identity_zero_vector():
    embeddings = {'Dhruvil':[[0.1]*512]}
    test_emb  = [0.0]*512
    result  = identity(test_emb=test_emb,embeddings=embeddings)
    assert result == "unknown"

def test_identity_nearest_person():
    np.random.seed(42)
    embeddings = {'Dhruvil':[np.random.rand(512)],
                  'keyur':[np.random.rand(512)]}
    test_emb  = embeddings['keyur'][0] + np.random.normal(0,0.01,512)
    result  = identity(test_emb=test_emb,embeddings=embeddings)
    assert result == 'keyur'
