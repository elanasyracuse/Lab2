import streamlit as st

st.set_page_config(page_title="🧪 Multipage Labs", page_icon="🧪", layout="centered")

# Define pages (each file is a standalone Streamlit script)
lab1 = st.Page("lab1.py", title="Lab 1 – Document QA", icon="📄")
lab2 = st.Page("lab2.py", title="Lab 2 – Document QA (default)", icon="📘", default=True)

# Create a navigation group and run
nav = st.navigation({"Labs": [lab1, lab2]})
nav.run()
