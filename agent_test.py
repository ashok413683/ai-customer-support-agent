from dotenv import load_dotenv
from langchain_groq import ChatGroq

from tools import create_support_ticket

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

llm_with_tools = llm.bind_tools([
    create_support_ticket
])

response = llm_with_tools.invoke(
    "My refund has not arrived. Please contact your support team."
)

print(response)