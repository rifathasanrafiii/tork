import requests
import streamlit as st

st.set_page_config(page_title="Tork AI", page_icon="🤖")
st.title("Tork AI")
st.caption("From-scratch mini GPT chat UI")

api_url = st.text_input("API URL", "http://127.0.0.1:8000/generate")
prompt = st.text_area("Prompt", "Ami ekta AI banate chai")
max_new_tokens = st.slider("Max new tokens", 10, 200, 80)

if st.button("Generate"):
    try:
        res = requests.post(api_url, json={"prompt": prompt, "max_new_tokens": max_new_tokens}, timeout=60)
        st.write(res.json())
    except Exception as exc:
        st.error(str(exc))
