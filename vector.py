from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        results = []
        for text in texts:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text
            )
            results.append(result.embeddings[0].values)
            time.sleep(0.7)
        return results

    def embed_query(self, text):
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

with open("alice_in_wonderland.md", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

documents = []
ids = []

for i, chunk in enumerate(chunks):
    document = Document(page_content=chunk, id=str(i))
    documents.append(document)
    ids.append(str(i))

embeddings = GeminiEmbeddings()

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="alice_in_wonderland",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)