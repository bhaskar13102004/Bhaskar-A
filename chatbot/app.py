import streamlit as st
import google.generativeai as genai

key = "AIzaSyC54XYAfH-V2wcswxDURqceqO_cCS5ukgw"
genai.configure(api_key=key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
chat = model.start_chat()

st.title("😶‍🌫️🤖AI chatbot!👾🧟‍♀️")
input = st.text_area("Enter your prompt here:")

if st.button("ASK!"):
    reply = chat.send_message(input)
    st.subheader("AI Response:")
    st.write(reply.text)
else:
    st.warning("Please enter a prompt!")