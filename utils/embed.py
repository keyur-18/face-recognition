from deepface import DeepFace
import os

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
                            detector_backend="yolov8n",
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
            json.dump(Embeddings,f)
    else :
        for person in os.listdir(persons):
            person_path = os.path.join(persons,person)
            Embeddings[person] = []
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path,img)
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



    

      
