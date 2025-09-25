import streamlit as st


st.set_page_config(page_title="🧪 Multipage Labs", page_icon="🧪", layout="centered")

lab1 = st.Page("lab1.py", title="Lab 1 Document QA", icon="📄")
lab2 = st.Page("lab2.py", title="Lab 2 Document QA (default)", icon="📘")
lab3 = st.Page("lab3.py", title="Lab 3 Chatbot using streamlit and OpenAI", icon="💬")
lab4 = st.Page("lab4.py", title="Lab 4 - Vector DB", icon="📒")
lab5 = st.Page("lab5.py", title="Lab 5 - The What To Wear Bot",icon="🤖",default=True)

nav = st.navigation({"Labs": [lab1, lab2, lab3, lab4, lab5]})
nav.run()
