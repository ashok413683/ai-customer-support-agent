from dotenv import load_dotenv
from langchain_groq import ChatGroq

from tools import create_support_ticket


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


tools = [
    create_support_ticket
]


llm_with_tools = llm.bind_tools(tools)