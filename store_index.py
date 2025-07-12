# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from dotenv import load_dotenv
# import os

# load_dotenv()  # ✅ Load variables from .env file

# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")


# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Delete old index (optional if starting fresh)
# pc.delete_index("medicalbot")


# index_name = "medicalbot"

# # ✅ Check if index exists before deleting
# if index_name in pc.list_indexes().names():
#     print("Deleting existing index...")
#     pc.delete_index(index_name)

# ✅ Recreate index
# print("Creating index...")
# pc.create_index(
#     name=index_name,
#     dimension=384,  # must match your embedding model
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# Recreate index with correct dimension

# pc.create_index(
#     name="medicalbot",
#     dimension=384,  # must match embedding model
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# Assuming you already have a PDF loader
# raw_docs = load_pdf_file("your-pdf-path.pdf")


# raw_docs = load_pdf_file("data/")

# text_chunks = text_split(raw_docs)  # returns chunked documents

# embeddings = download_hugging_face_embeddings()  # returns embedding model
# index_name = "medicalbot"  # define index name


# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings,
# )

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()  # ✅ Load variables from .env file

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# ✅ Check if index exists before deleting
if index_name in pc.list_indexes().names():
    print("Deleting existing index...")
    pc.delete_index(index_name)

# ✅ Create new index
print("Creating index...")
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# ✅ Load PDF documents from folder
raw_docs = load_pdf_file("data/")  # Assuming you placed your PDFs in a 'data/' folder
text_chunks = text_split(raw_docs)

# ✅ Get embeddings and store them in Pinecone
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
