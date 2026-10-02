from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

memory = MemorySaver()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {'messages': [response]}


graph = StateGraph(ChatState)

graph.add_edge(START, "chat_node")
graph.add_node("chat_node", chat_node)
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=memory)

thread_id = "demo-thread"

while True:

    user_input = input("User: ")

    if user_input.lower().strip() in {"exit", "quit"}:
        print("Exiting the chat. Goodbye!")
        break

    state = {"messages": [HumanMessage(content=user_input)]}

    # Run the graph
    result = chatbot.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )

    # Get the latest message from the assistant
    messages = result["messages"]
    last_msg = messages[-1]
    print(f"Bot: {last_msg.content}\n")

# ---  TO SEE THE GRAPH IN POWERSHELL ---
# chatbot = graph.compile()
# chatbot.get_graph().print_ascii()
