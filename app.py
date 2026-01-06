import streamlit as st

face_enroll = st.Page("pages/face_enroll.py",title="face enroll")
face_recog = st.Page("pages/face_recog.py",title="face recognition")

pg = st.navigation([face_enroll,face_recog])

st.set_page_config(page_title="Face recognition app")

pg.run()