from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

app = Flask(__name__)
client = OpenAI()

# ===========================
# 🔥 HARDCODE THESE VALUES
# ===========================
VECTOR_STORE_ID = "vs_6915c487e8ac81919702f97efc7284e7"
ASSISTANT_ID    = "asst_z8QTN82CqFMX1RXqKMaEOdph"


# ==========================================
# 1. UPLOAD CSV → Upload file to vector store
# ==========================================
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "CSV file missing"}), 400

    file_storage = request.files["file"]

    temp = tempfile.NamedTemporaryFile(delete=False)
    file_storage.save(temp.name)
    real_file = open(temp.name, "rb")

    # Upload file into correct vector store
    client.vector_stores.files.upload_and_poll(
        vector_store_id=VECTOR_STORE_ID,
        file=real_file
    )

    real_file.close()

    return jsonify({
        "message": "CSV uploaded & embedded successfully",
        "vector_store_id": VECTOR_STORE_ID
    })


# ==========================================
# 2. CREATE ASSISTANT LINKED TO VECTOR STORE
# ==========================================
@app.route("/create_assistant", methods=["POST"])
def create_assistant():

    assistant = client.beta.assistants.create(
        name="CSV Query Bot",
        model="gpt-4.1-mini",
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {"vector_store_ids": [VECTOR_STORE_ID]}
        }
    )

    return jsonify({
        "message": "Assistant created successfully",
        "assistant_id": assistant.id
    })


# ==========================================
# 3. ASK QUESTION
# ==========================================

def format_results(results):
    formatted_results = ''
    for result in results.data:
        formatted_result = f"<result file_id='{result.file_id}' file_name='{result.file_name}'>"
        for part in result.content:
            formatted_result += f"<content>{part.text}</content>"
        formatted_results += formatted_result + "</result>"
    return f"<sources>{formatted_results}</sources>"

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("question")

    # 1️⃣ Vector search
    results = client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=query
    )

    # 2️⃣ Extract text chunks from results
    chunks = []
    for r in results.data:
        if r.content:
            chunks.append(r.content)

    context_text = "\n\n".join(chunks)

    # 3️⃣ LLM Completion (RAG)
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "developer",
                "content": "Answer the question concisely using ONLY the provided context. If the answer is not in the context, say 'Not found in data.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}"
            }
        ]
    )

    answer = completion.choices[0].message.content

    return jsonify({
        "question": query,
        "answer": answer,
        "chunks_used": chunks
    })



if __name__ == "__main__":
    app.run(debug=True)
