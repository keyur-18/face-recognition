import streamlit as st
from deepface import DeepFace
from scipy.spatial.distance import cosine
from utils.embed import save_embeddings,recognition

st.set_page_config(page_title="Face Recognition")
st.title("Face Recognition")
img=st.camera_input("capture your image")


generate = st.button("Generate")

if generate:
    if img is not None:
        name = recognition(img)
        st.success(f"Recognized as: {name}")
    else:
        st.warning("Please capture an image")
    st.balloons()
