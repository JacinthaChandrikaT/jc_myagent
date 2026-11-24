
import streamlit as st
import joblib
st.set_page_config(
    page_title="My Python AI Agent",
    page_icon="🤖"
)

st.title("🤖 My Python AI Agent")
st.write("Ask your AI Agent a question")

# Replace with your real agent function
def ask_agent(question):
    return f"Your agent received: {question}"

question = st.chat_input("Enter your question")

if question:
    with st.chat_message("user"):
        st.write(question)

    #answer = ask_agent(question)

    with st.chat_message("assistant"):
        #st.write(answer)
        model = joblib.load("my_model.joblib")
        answer = model.predict([question])[0]
        st.write(answer)
