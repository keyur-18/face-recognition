import streamlit as st
# from deepface import DeepFace
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from utils.embed import save_embeddings,recognition
from ultralytics  import YOLO
model = YOLO(r'model\yolov8n-face.pt')
st.set_page_config(page_title="Face Recognition")
st.title("Face Recognition")
img=st.camera_input("capture your image")


generate = st.button("Generate")

if generate:
    if img is not None:
        ans = []
        img = Image.open(img)
        img = np.array(img)
        results = model.track(img)
        boxes = results[0].boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            face = img[y1:y2,x1:x2]
            name = recognition(face)
            ans.append(name)
        st.success(f"Recognized as: {ans}")
    else:
        st.warning("Please capture an image")
    st.balloons()
