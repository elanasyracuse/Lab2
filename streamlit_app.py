import streamlit as st

st.set_page_config(page_title="🧪 Multipage Labs", page_icon="🧪", layout="centered")

lab1 = st.Page("lab1.py", title="Lab 1 Document QA", icon="📄")
lab2 = st.Page("lab2.py", title="Lab 2 Document QA (default)", icon="📘")
lab3 = st.Page("lab3.py", title="Lab 3 Chatbot using streamlit and OpenAI", icon="💬")
lab4 = st.Page("lab4.py", title="Lab 4 - Vector DB", icon="📒")
lab5 = st.Page("lab5.py", title="Lab 5 - The What To Wear Bot",icon="🤖")
lab8 = st.Page("lab8.py",icon="📒", title="Lab 8 ")
lab9 = st.Page("lab9.py",icon="📒", title="Lab 9")  # Changed st.page to st.Page
presentationlab4 = st.Page("presentationlab4.py", title="AI Fact-Checker + Citation Builder", icon="🤖")
research_project = st.Page("researchproject.py", title="Weekly Emails ",icon="📒", default=True)
standalone_project = st.Page("standalone.py", title="standalone ",icon="📒")

nav = st.navigation({"Labs": [lab1, lab2, lab3, lab4, lab5, lab8, lab9, presentationlab4, research_project, standalone_project]})
nav.run()