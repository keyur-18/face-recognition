import streamlit as st
from utils.embed import save_embeddings
from utils.enroll_utils import image_validation,save_images,creat_user_dir
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
    if not image_validation(images):
        st.error("please capure images")
    else:
        user_dir = creat_user_dir("images",username)
        save_images(images,user_dir)
 
        # augmentation(username)
        save_embeddings(username)
        st.success("images saved")
        st.balloons()
