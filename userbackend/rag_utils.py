import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Batch
from qdrant_client.models import QueryRequest



# Load environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Qdrant client
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "poc_collection"

# Ensure collection exists
def init_qdrant():
    collections = qdrant.get_collections()

    existing = [c.name for c in collections.collections]

    if COLLECTION_NAME in existing:
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists -> skipping creation.")
        return

    print(f"🚀 Creating collection '{COLLECTION_NAME}'...")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )
    
# Extract text
def extract_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages   # list of Document objects

# Chunk text
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

# Store embeddings
def store_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectors = []
    payloads = []

    ids = []
    idx = 1

    for c in chunks:
        vec = embeddings.embed_query(c.page_content)
        vectors.append(vec)
        payloads.append({"text": c.page_content})
        ids.append(idx)
        idx += 1

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=Batch(ids=ids, vectors=vectors, payloads=payloads)
    )


def retrieve_context(query):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vec = embeddings.embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=4
    )

    context_text = "\n".join([point.payload.get("text", "") for point in results.points])
    return context_text

