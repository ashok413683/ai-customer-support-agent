from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from tools import create_support_ticket


load_dotenv()


app = FastAPI(
    title="AI Customer Support Agent"
)

conversation_memory = {}

# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# -------------------------
# Tool
# -------------------------

tools = [
    create_support_ticket
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------
# Embeddings
# -------------------------

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b"
)


# -------------------------
# Chroma
# -------------------------

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# -------------------------
# Retriever
# -------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -------------------------
# Request Model
# -------------------------

class ChatRequest(BaseModel):
    session_id: str
    name: str
    email: str
    message: str


# -------------------------
# Home
# -------------------------

@app.get("/")
def home():

    return {
        "message": "AI Customer Support Agent is running"
    }


# -------------------------
# Chat API
# -------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    history = conversation_memory.get(
    request.session_id,
    []
   )
    
    # 1. Retrieve relevant information
    documents = retriever.invoke(
        request.message
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # 2. System instructions

    system_message = SystemMessage(
    content=f"""
You are a professional customer support AI agent.

Customer name:
{request.name}

Customer email:
{request.email}

Knowledge Base:
{context}


IMPORTANT RULES:

1. Use the knowledge base to answer the customer.

2. Never invent company policies.

3. If the customer's issue can be solved directly
   using the knowledge base, answer the customer.

4. If the customer reports that a refund is delayed
   beyond the stated processing time, create a support ticket.

5. The refund processing time is 5-7 business days.

6. If the customer says their refund is older than
   7 business days, DO NOT ask for more information first.
   Create a support ticket immediately.

7. When creating a support ticket, use:
   customer_name = "{request.name}"
   customer_email = "{request.email}"
   issue = the customer's complete issue.

8. Be polite and concise.
"""
)

    # 3. Customer message

    human_message = HumanMessage(
        content=request.message
    )


    messages = [
        system_message
    ]

    messages.extend(history)
    messages.append(human_message)
    
    conversation_memory[request.session_id] = history
    
    # 4. Ask LLM

    response = llm_with_tools.invoke(
        messages
    )


    # 5. Check if AI wants to use a tool

    if response.tool_calls:

        for tool_call in response.tool_calls:

            if tool_call["name"] == "create_support_ticket":

                tool_result = create_support_ticket.invoke(
                    tool_call["args"]
                )


                # Add AI tool call
                messages.append(response)


                # Add tool result
                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                )


                # 6. Ask LLM for final answer

                final_response = llm.invoke(
                    messages
                )
                history.append(
                        AIMessage(content=final_response.content)
                    )


                return {
                    "answer": final_response.content,
                    "ticket_created": True,
                    "sources": [
                        document.page_content
                        for document in documents
                    ]
                }


    # No tool required

    return {
        "answer": response.content,
        "ticket_created": False,
        "sources": [
            document.page_content
            for document in documents
        ]
    }