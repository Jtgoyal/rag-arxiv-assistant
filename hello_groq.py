# hello_groq.py
# Day 1 deliverable: prove that our environment + API key work
# by making a real LLM call to Groq's Llama 3.

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Step 1: Load environment variables from .env into os.environ
# After this line, GROQ_API_KEY is available to any library that looks for it.
load_dotenv()

# Step 2: Create an LLM client.
# ChatGroq is LangChain's wrapper around Groq's API.
# It automatically reads GROQ_API_KEY from the environment.
# We pick "llama-3.1-8b-instant" — small, fast, free-tier-friendly.
llm = ChatGroq(model="llama-3.1-8b-instant")

# Step 3: Send a message.
# HumanMessage is LangChain's standard way to represent a user message.
# The invoke() method sends it to the LLM and returns the response.
response = llm.invoke([HumanMessage(content="Explain Retrieval-Augmented Generation (RAG) in AI in 3 sentences.")])
# Step 4: Print just the text content of the response.
# response is an AIMessage object; .content gives us the actual text.
print(response.content)