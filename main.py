from google import genai
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from vector import retriever
import os

load_dotenv()

client = genai.Client(api_key="AIzaSyBQc-SeqdX921kuNGAkzrwdiIjvGXOz-SI")

template = """You are an expert on the book "Alice in Wonderland."

Below are relevant chapters extracted from the book to help answer the question.

Relevant chapters:
{chapters}

Question:
{question}

Provide a clear, concise summary answer now:
"""

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    chapters = retriever.invoke(question)
    
    prompt = template.format(chapters=chapters, question=question)
    
    response = client.models.generate_content(
        model="models/gemini-1.5-flash",
        contents=prompt
    )
    print(response.text)