from deepface import DeepFace
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cosine
import numpy as np
from PIL import Image
import cv2
import json
path = "../images"

def create_embeddings(img_path):
    res = DeepFace.represent(img_path=img_path,
                             model_name="ArcFace",
                            detector_backend="yolov8n",
                            enforce_detection=False)
    vec= res[0]['embedding']  
    return vec

def load_db(path="face_embeddings.json"):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def save_db(db, path="face_embeddings.json"):
    with open(path, "w") as f:
        json.dump(db, f)

def save_embeddings(name):
    data = load_db()
    data.setdefault(name, [])

    for img in os.listdir(f"images/{name}/all"):
        img_path = f"images/{name}/all/{img}"
        embed = create_embeddings(img_path)
        data[name].append(embed)
    save_db(data)

def is_zero_vector(v, eps=1e-6):
    return np.linalg.norm(v) < eps
def safe_cosine(a, b):
    if is_zero_vector(a) or is_zero_vector(b):
        return 1.0
    return cosine(a, b)
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v /norm

def identity(test_emb,embeddings,THRESHOLD=0.5,MARGIN = 0.05):
    test_emb = normalize(test_emb)
    if is_zero_vector(test_emb):
        return "unknown"
    best_person = "unknown"
    min_dist = 1.0
    second_best = 1.0
    for person, embs in embeddings.items():
        for known_emb in embs:
            known_emb  =normalize(known_emb)
            dist = safe_cosine(test_emb, known_emb)
            if dist < min_dist:
                second_best = min_dist
                min_dist = dist
                best_person = person
            elif min_dist<second_best:
                second_best = min_dist
    if min_dist >THRESHOLD or (second_best-min_dist) <MARGIN:
        return "unknown"
    return best_person
def recognition(img):
    pil_img = Image.open(img)
    img_arr = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_arr,cv2.COLOR_RGB2BGR)
    embeddings = load_db()
    emb  = create_embeddings(img_bgr)
    return identity(emb,embeddings=embeddings)


def augmentation(name):
    datagen = ImageDataGenerator(rotation_range = 30,
                                 shear_range = 0.2,
                                 width_shift_range = 0.2,
                                 height_shift_range = 0.2,
                                 brightness_range = [0.5,1.5])
    cnt = 0
    for img in datagen.flow_from_directory(f"images/{name}",batch_size=1,save_to_dir = f"images/{name}/all"):
        cnt = cnt+1
        if cnt==10:
            break
    

    

      
