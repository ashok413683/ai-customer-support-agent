import streamlit as st
import requests

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖"
)

st.title("🤖 AI Customer Support Agent")
st.write("Ask your question and our AI assistant will help you.")

# Customer details
name = st.text_input("Your Name")
email = st.text_input("Your Email")

# Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = "ashok-001"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
user_message = st.chat_input("Type your question...")

if user_message:

    if not name or not email:
        st.warning("Please enter your name and email first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.write(user_message)

    # Send request to FastAPI
    payload = {
        "session_id": st.session_state.session_id,
        "name": name,
        "email": email,
        "message": user_message
    }

    try:

        response = requests.post(
            "https://ai-customer-support-agent-g3q1.onrender.com/chat",
            json=payload
        )

        if response.status_code == 200:

            data = response.json()

            answer = data["answer"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.write(answer)

            if data.get("ticket_created"):
                st.success("🎫 Support ticket created successfully!")

        else:
            st.error("Something went wrong with the AI server.")

    except Exception as e:

        st.error(f"Could not connect to FastAPI: {e}")