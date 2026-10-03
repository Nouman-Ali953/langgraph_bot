import streamlit as st

from chatbot import chatbot
from chatbot import retrieve_all_threads

from langchain_core.messages import HumanMessage

import uuid


st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
)


st.title("🤖 LangGraph Chatbot")

st.caption("Chatbot powered by LangGraph + Gemini")


# ************************************************************
# Utility Functions
# ************************************************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(st.session_state["thread_id"])

    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Check if messages key exists in state values
    # Return empty list if not
    return state.values.get("messages", [])


# ************************************************************
# Session Setup
# ************************************************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


add_thread(st.session_state["thread_id"])


# ************************************************************
# Sidebar UI
# ************************************************************

st.sidebar.title("LangGraph Chatbot")


# New Chat button
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


# Conversations
st.sidebar.header("My Conversations")


for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(str(thread_id)):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            # Gemini content can sometimes be a list of content blocks.
            # Convert it to a string for Streamlit if necessary.
            content = msg.content

            if isinstance(content, list):

                text_content = ""

                for block in content:

                    if isinstance(block, dict):

                        text = block.get("text")

                        if text:
                            text_content += text

                    elif isinstance(block, str):

                        text_content += block

                content = text_content

            temp_messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

        st.session_state["message_history"] = temp_messages

        st.rerun()


# ************************************************************
# Main UI
# ************************************************************

# Display previous messages

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


user_input = st.chat_input("Type here ...")


if user_input:

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(user_input)


    # --------------------------------------------------------
    # LangGraph Configuration
    # --------------------------------------------------------

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },

        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },

        "run_name": "chat_turn",
    }


    # --------------------------------------------------------
    # Streaming Response
    # --------------------------------------------------------

    def stream_response():

        for message_chunk, metadata in chatbot.stream(

            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },

            config=CONFIG,

            stream_mode="messages"

        ):

            content = message_chunk.content


            # Gemini sometimes returns plain text

            if isinstance(content, str):

                yield content


            # Gemini can also return structured content blocks

            elif isinstance(content, list):

                for block in content:

                    if isinstance(block, dict):

                        text = block.get("text")

                        if text:

                            yield text


    # --------------------------------------------------------
    # Display Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            ai_message = st.write_stream(
                stream_response()
            )


    # --------------------------------------------------------
    # Save Assistant Response
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )