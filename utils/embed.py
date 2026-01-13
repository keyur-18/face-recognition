from deepface import DeepFace
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cosine
import numpy as np
from PIL import Image
import streamlit as st
import cv2
import json
path = "../images"

def create_embeddings(img_path):
    res = DeepFace.represent(img_path=img_path,
                             model_name="ArcFace",
                            detector_backend="retinaface",
                            enforce_detection=False)
    vec= res[0]['embedding']  
    return vec


def save_embeddings(name):
    Embeddings = {}
    persons = "images"
    if "face_embeddings.json" in os.listdir():
        Embeddings[name] = []
        for img in os.listdir(f"images/{name}"):
            img_path = f"images/{name}/{img}"
            print(img_path)
            embed = create_embeddings(img_path)
            Embeddings[name].append(embed)
            with open("face_embeddings.json","r") as f:
                data = json.load(f)
            data.update(Embeddings)
        with open("face_embeddings.json","w") as f:
            json.dump(data,f)
    else :
        for person in os.listdir(persons):
            person_path = os.path.join(persons,person)
            Embeddings[person] = []
            for img in os.listdir(f"{person_path}"):
                img_path = f"{person_path}/{img}"
                print(img_path)
                embed = create_embeddings(img_path)
                Embeddings[person].append(embed)
        with open("face_embeddings.json","w")as f:
            json.dump(Embeddings,f)


def recognition(img):
    pil_img = Image.open(img)
    img_arr = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_arr,cv2.COLOR_RGB2BGR)
    THRESHOLD = 0.5 
    with open("face_embeddings.json", "r") as f:
        embeddings = json.load(f)
    emb  = create_embeddings(img_bgr)
    name = "unknown"
    min_dist = 1.0
    for person, embs in embeddings.items():
        for known_emb in embs:
            dist = cosine(emb, known_emb)
            if dist < min_dist:
                min_dist = dist
                best_person = person
    if min_dist >THRESHOLD:
        best_person = "unknown"
    return best_person


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
    

    

      
