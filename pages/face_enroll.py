import streamlit as st
from PIL import Image
import os
import io
from datetime import datetime
from utils.embed import save_embeddings
st.set_page_config(page_title="Face enroll",layout="centered")
st.title("face enrollment")
st.write("Capture **3 face images** from different angles")


username = st.text_input("enter username")
if not username:
    st.warning("Please enter username to continue")
    st.stop()
st.subheader("capure images")

img_front = st.camera_input("Front face")
img_left  = st.camera_input("Left Angle")
img_right = st.camera_input("Right Angle")
# print(type(img_front))
images = {
    "front":img_front,
    "left":img_left,
    "right":img_right
}


if st.button("save images"):
    if None in images.values():
        st.error("please capure images")
    else:
        user_dir = f"images/{username}"
        os.makedirs(user_dir,exist_ok=True)

        for angle,img in images.items():
            image = Image.open(img)
            image.save(f"{user_dir}/{angle}.jpg")
        save_embeddings()
        st.success("images saved")
        st.balloons()
