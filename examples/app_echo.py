"""Zhang Wei's first working UI, kept for later learning exercises."""

import streamlit as st

st.title("QuantRAG")
query = st.text_input("Enter your question")
if st.button("Submit"):
    st.write(query)
