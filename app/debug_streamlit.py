"""Minimal Streamlit debug app to test rendering."""

import streamlit as st

st.set_page_config(page_title="Debug", page_icon="🐛", layout="centered")

st.success("✅ Debug app loaded successfully!")
st.write("If you see this message, Streamlit rendering works.")

st.text_input("Test input field", value="Type something here")
st.button("Test button", type="primary")

st.info("✅ All basic Streamlit components are rendering.")
