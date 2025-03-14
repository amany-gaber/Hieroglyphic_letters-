import os
import streamlit as st

st.write("📂 Current Directory:", os.getcwd())
st.write("📄 Files in Directory:", os.listdir())

if "hieroglyphic_model.pt" not in os.listdir():
    st.error("⚠️ Model file 'hieroglyphic_model.pt' not found! Make sure it's uploaded correctly.")
