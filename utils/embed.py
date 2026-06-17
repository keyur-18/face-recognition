# from deepface import DeepFace
import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cosine
import numpy as np
import json
import onnxruntime as ort
import cv2
from PIL import Image
from ultralytics import YOLO
model = YOLO(r'model\yolov8n-face.pt')
session = ort.InferenceSession(r'model\arc.onnx')
print(session.get_inputs()[0].shape)
def create_embeddings(img):
    face = cv2.resize(img,(112,112))
    face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)

    face = (face-127.5)/128
    face = np.expand_dims(face , axis = 0)
    input_name = session.get_inputs()[0].name
    embedding = session.run(None,{input_name : face})[0][0]
    embedding = embedding/np.linalg.norm(embedding)
    # res = DeepFace.represent(img_path=img_path,
    #                          model_name="ArcFace",
    #                         detector_backend="retinaface",
    #                         enforce_detection=False)
    # vec= res[0]['embedding']  
    return embedding

def load_db(path="face_embeddings.json"):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def save_db(db, path="face_embeddings.json"):
    with open(path, "w") as f:
        json.dump(db, f)

def save_embeddings(name,imgs):
    data = load_db()
    data.setdefault(name, [])

    for img in imgs:
        img = Image.open(img)
        img = np.array(img)
        results = model.track(img)
        boxes = results[0].boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            face = img[y1:y2,x1:x2]
        embed = create_embeddings(face)
        data[name].append(embed.tolist())
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
    THRESHOLD = 0.7
    test_emb = normalize(test_emb)
    if is_zero_vector(test_emb):
        return "unknown"
    best_person = "unknown"
    best_score = -1.0
    # second_best = 1.0
    for person, embs in embeddings.items():
        for known_emb in embs:
            known_emb = np.array(known_emb, dtype=np.float32)
            known_emb  =normalize(known_emb)
            score = np.dot(test_emb, known_emb)
            if score > best_score:
                best_score = score
                best_person = person

    print("Best score:", best_score)

    if best_score <THRESHOLD :
        return "unknown"
    return best_person
def recognition(img):
    embeddings = load_db()
    emb  = create_embeddings(img)
    return identity(emb,embeddings=embeddings)


# def augmentation(name):
#     datagen = ImageDataGenerator(rotation_range = 30,
#                                  shear_range = 0.2,
#                                  width_shift_range = 0.2,
#                                  height_shift_range = 0.2,
#                                  brightness_range = [0.5,1.5])
#     cnt = 0
#     for img in datagen.flow_from_directory(f"images/{name}",batch_size=1,save_to_dir = f"images/{name}"):
#         cnt = cnt+1
#         if cnt==10:
#             break
    

    
###### this is for fastapi

def register_user(name,path):
    data = load_db()
    data.setdefault(name, [])

    for img in os.listdir(path):
        img_path = f"{path}/{img}"
        embed = create_embeddings(img_path)
        data[name].append(embed)
    save_db(data)
