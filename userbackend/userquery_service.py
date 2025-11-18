from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from rag_utils import (
    init_qdrant, extract_pdf_text, chunk_documents, store_embeddings, retrieve_context
)
import os
from dotenv import load_dotenv


load_dotenv()


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Initialize Qdrant on server start
init_qdrant()

# ---------------------------------------------------
# 👉 1. Upload PDF → create embeddings
# ---------------------------------------------------
@app.post("/upload_pdf")
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "PDF file is required"}), 400

    pdf = request.files["pdf"]
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(pdf_path)

    # Extract → Chunk → Store
    docs = extract_pdf_text(pdf_path)
    chunks = chunk_documents(docs)
    store_embeddings(chunks)
    os.remove(pdf_path)

    return jsonify({"message": "PDF processed and embeddings stored."})


@app.post("/userquery")
def userquery():
    data = request.json
    question = data.get("query", "").strip()

    if not question:
        return jsonify({"error": "query field is required"}), 400

    # Retrieve relevant context
    context = retrieve_context(question)

    if not context:
        return jsonify({"answer": "No relevant context found."})

    # Build final prompt
    system_prompt = f"""
Answer strictly using the provided PDF content.
If answer is not found, reply: "Information not found in context."

Context:
{context}
"""

    answer = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]).content

    return jsonify({
        "question": question,
        "answer": answer,
        "context_used": context
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
