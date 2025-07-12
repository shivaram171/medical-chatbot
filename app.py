from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_openai import OpenAI
from langchain_community.llms import Together
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.prompt import *
import os

app= Flask(__name__)



load_dotenv()

# PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
# TOGETHER_API_KEY=os.environ.get("TOGETHER_API_KEY")


# import os
# from dotenv import load_dotenv  # ✅ Import this BEFORE using load_dotenv()

# load_dotenv()  # ✅ Load variables from .env file

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")


embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

#from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=512,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    
    response = rag_chain.invoke({"input": msg})
    raw_answer = response["answer"]
    print("Raw model response:", raw_answer)

    # Clean the output: remove `? Assistant:` and `Human:` if present
    cleaned_answer = raw_answer.replace("? Assistant:", "").replace("Human:", "").strip()
    
    return cleaned_answer


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     print("User input:", msg)
    
#     # Invoke the chain
#     response = rag_chain.invoke({"input": msg})
#     raw_answer = str(response.get("answer", ""))
#     print("Raw model response:", raw_answer)

#     # Clean unwanted prefixes from the start
#     prefixes_to_remove = ["? assistant:", "assistant:", "human:", "?"]
#     cleaned_answer = raw_answer.strip()
#     for prefix in prefixes_to_remove:
#         if cleaned_answer.lower().startswith(prefix):
#             cleaned_answer = cleaned_answer[len(prefix):].strip()

#     return cleaned_answer





if __name__ == "__main__":
    app.run(debug=True, port=5000)
