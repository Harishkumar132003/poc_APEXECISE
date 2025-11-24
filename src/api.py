from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import os
from sqlalchemy import create_engine, text
from flask_cors import CORS
from openai import OpenAI
import tempfile
import base64
import json

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
client = OpenAI()

# ---------------------------------------------------------------
# DB ENGINE
# ---------------------------------------------------------------

raw_engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:3306/{os.getenv('DB_NAME')}",
    pool_pre_ping=True,
)


def save_chat(usercode, role, message=None, response=None, audio_blob=None):
    try:
        query = text(
            """
            INSERT INTO chat_history (usercode, role, message, audio, response)
            VALUES (:usercode, :role, :message, :audio, :response)
        """
        )
        with raw_engine.begin() as conn:
            conn.execute(
                query,
                {
                    "usercode": usercode,
                    "role": role,
                    "message": message,
                    "audio": audio_blob,
                    "response": response,
                },
            )
    except Exception as e:
        print("Chat save error:", e)


def get_chat_by_usercode(usercode):
    query = text(
        """
        SELECT role, message, audio, response, created_at
        FROM chat_history
        WHERE usercode = :usercode
        ORDER BY id ASC
    """
    )

    with raw_engine.begin() as conn:
        rows = conn.execute(query, {"usercode": usercode}).mappings().all()

    result = []
    for r in rows:
        audio_base64 = (
            f"data:audio/webm;base64,{base64.b64encode(r['audio']).decode()}"
            if r["audio"]
            else None
        )
        parsed_response = None
        try:
            parsed_response = json.loads(r["response"])
        except:
            parsed_response = r["response"]

        result.append(
            {
                "role": r["role"],
                "message": r["message"],
                "response": parsed_response,
                "created_at": r["created_at"].isoformat(),
                "audio": audio_base64,
            }
        )

    return result


def init_database():
    db_uri = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:3306/{os.getenv('DB_NAME')}"

    engine_args = {"pool_pre_ping": True, "pool_size": 20, "max_overflow": 40}

    return SQLDatabase.from_uri(db_uri, engine_args=engine_args)


db = init_database()

# Cache schema
SCHEMA = db.get_table_info()

# ---------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------

SYSTEM_SQL_ANALYST = f"""
You are an expert MySQL analyst.
You know the following database schema:

<SCHEMA>
{SCHEMA}
</SCHEMA>

BUSINESS FLOW:
distillery → wholesale (DEPO) → retail → customer

IMPORTANT LOGIC (STRICT):

ROLE RULES (OVERRIDES EVERYTHING):
- Role = "depot":
      • You MUST use ONLY poc_wholesale and poc_stock_closing
      • NEVER use poc_distillery
      • NEVER use poc_retail

- Role = "distillery":
      • You MUST use ONLY poc_distillery and poc_stock_closing
      • NEVER use poc_wholesale
      • NEVER use poc_retail

USER TYPES:
   A. Depot user (USERCODE starts with 'DEPO'):
        - Use depot logic ONLY IF role is not provided.
        - Depot outgoing dispatch → poc_wholesale.from_entity_code = '<USERCODE>'
        - Closing stock → poc_stock_closing.entity_code = '<USERCODE>'

   B. Distillery user (USERCODE does NOT start with 'DEPO'):
        - Use distillery logic ONLY IF role is not provided.
        - Distillery outgoing dispatch → poc_distillery.from_entity_code = '<USERCODE>'
        - Closing stock → poc_stock_closing.entity_code = '<USERCODE>'

FOR DEPOT USERS (USERCODE starts with 'DEPO'):
   ALWAYS use:
       FROM poc_wholesale
       WHERE from_entity_code = '<USERCODE>'

FOR DISTILLERY USERS (USERCODE does NOT start with 'DEPO'):
   ALWAYS use:
         FROM poc_distillery
         WHERE from_entity_code = '<USERCODE>'

SQL RULES:
- Output ONLY raw SQL.
- No markdown.
- No explanation.
"""

SYSTEM_DATA_ANALYST = """
You are a senior data analyst.
Rules:
- Interpret the SQL result into a short, clear answer.
- Never mention SQL.
- Never mention schema.
"""

# ---------------------------------------------------------------
# LLM MODELS
# ---------------------------------------------------------------

llm_sql = ChatOpenAI(model="gpt-4.1", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_chart = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------


def clean_sql(q):
    return q.replace("```sql", "").replace("```", "").strip()


def wants_chart(user_query: str):
    chart_keywords = [
        "chart",
        "graph",
        "plot",
        "visualize",
        "trend",
        "bar",
        "line",
        "pie",
    ]
    return any(k in user_query.lower() for k in chart_keywords)


def generate_chart_json(sql_results, question):
    """
    Ask the LLM for a chart spec and return a CLEAN dict.

    On success: returns only chart fields:
        { "chart_type", "labels", "values"/"datasets", "title" }

    On failure: returns a TEXT-style object:
        { "type": "text", "response": "..." }

    So process_query can safely do:
        { "type": "chart", **chart_json }
        and let chart_json override to text if it failed.
    """
    prompt = f"""
Convert SQL result into a chart-ready JSON.

User question: "{question}"

Rules:
- STRICT JSON only.
- NO markdown.
- NO backticks.
- Use DOUBLE QUOTES everywhere.
- Chart must ALWAYS include colors.

Chart format for single-series:
{{
  "chart_type": "pie" | "bar" | "line",
  "labels": [...],
  "values": [...],
  "colors": ["#RRGGBB", "#RRGGBB", ...],
  "title": "..."
}}

Chart format for multi-series:
{{
  "chart_type": "bar" | "line",
  "labels": [...],
  "datasets": [
    {{
      "label": "Series name",
      "data": [...],
      "color": "#RRGGBB"
    }}
  ],
  "title": "..."
}}

Color Rules:
- Use bright modern colors.
- Do not repeat colors in the same chart.
- Use hex format (e.g., "#3b82f6").

SQL Result:
{sql_results}
"""

    raw = llm_chart.invoke(prompt).content

    # Strip code fences if model still adds them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)

        # Basic validation
        if (
            "chart_type" not in parsed
            or "labels" not in parsed
            or "title" not in parsed
        ):
            raise ValueError("Missing required chart keys")

        # Normalize structure
        chart_obj = {
            "chart_type": parsed["chart_type"],
            "labels": parsed["labels"],
            "title": parsed["title"],
        }
        if "datasets" in parsed:
            chart_obj["datasets"] = parsed["datasets"]
        else:
            chart_obj["values"] = parsed.get("values", [])
            chart_obj["colors"] = parsed.get("colors", [])

        return chart_obj

    except Exception as e:
        # Fallback: treat as a normal text response instead of broken chart
        return {
            "type": "text",
            "response": "I could not generate a valid chart for this query, but I can still answer in text if you ask again without requesting a chart.",
        }


# ---------------------------------------------------------------
# MAIN PROCESSING PIPELINE
# ---------------------------------------------------------------


def generate_sql(user_question: str, usercode: str, role: str):
    system_prompt = SYSTEM_SQL_ANALYST.replace("<USERCODE>", usercode).replace(
        "<ROLE>", role
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    sql = llm_sql.invoke(messages).content
    return clean_sql(sql)


def generate_final_answer(question, sql_query, sql_results, chat_history):
    messages = [
        {"role": "system", "content": SYSTEM_DATA_ANALYST},
        *chat_history,
        {
            "role": "user",
            "content": f"""
User Question: {question}
SQL Result: {sql_results}
""",
        },
    ]
    return llm_answer.invoke(messages).content


def process_query(user_query, usercode, role, chat_history):

    sql = generate_sql(user_query, usercode, role)

    try:
        sql_result = db.run(sql)
    except Exception as e:
        return {"type": "text", "response": f"SQL Error: {e}\nGenerated SQL: {sql}"}

    # If user wants chart
    if wants_chart(user_query):
        chart_json = generate_chart_json(sql_result, user_query)
        # If chart_json failed, it returns type="text"
        return {"type": "chart", **chart_json}

    # Otherwise return text
    history_slice = chat_history[-2:]
    text_answer = generate_final_answer(user_query, sql, sql_result, history_slice)

    return {"type": "text", "response": text_answer}


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------

chat_history = [AIMessage(content="Hello! I'm your SQL assistant. Ask me anything.")]


@app.post("/analyze")
def analyze():
    data = request.json

    user_query = data.get("query")
    usercode = data.get("usercode", "")
    role = data.get("role", "").lower().strip()

    if not user_query:
        return jsonify({"error": "query field is required"}), 400

    chat_history.append(HumanMessage(content=user_query))

    try:
        result = process_query(user_query, usercode, role, chat_history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # For LLM conversation history, store text only (for context)
    history_text = (
        result.get("response")
        if isinstance(result, dict) and result.get("type") == "text"
        else "Chart response generated."
    )
    chat_history.append(AIMessage(content=history_text))

    # Store FULL JSON in DB so frontend can reconstruct chart/text
    try:
        save_chat(
            usercode, "assistant", message=user_query, response=json.dumps(result)
        )
    except Exception as e:
        print("Error saving chat:", e)

    # Return pure JSON to frontend
    return jsonify(result)


@app.get("/analyze/history/<usercode>")
def analyze_history(usercode):
    results = get_chat_by_usercode(usercode)
    return jsonify({"usercode": usercode, "history": results})


@app.post("/voice")
def voice_input():
    try:
        audio_file = request.files.get("audio")
        usercode = request.form.get("usercode", "")
        role = request.form.get("role", "").lower().strip()

        if not audio_file:
            return jsonify({"error": "Audio file is required"}), 400

        audio_bytes = audio_file.read()
        audio_file.stream.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
            audio_file.save(temp.name)
            audio_path = temp.name

        # Whisper transcription
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", file=f, language="en"
            )

        transcribed_text = transcript.text.strip()

        reply = process_query(transcribed_text, usercode, role, chat_history)

        # For history context:
        history_text = (
            reply.get("response")
            if isinstance(reply, dict) and reply.get("type") == "text"
            else "Chart response generated."
        )

        chat_history.append(HumanMessage(content=transcribed_text))
        chat_history.append(AIMessage(content=history_text))

        # Save to DB (store reply as JSON string)
        save_chat(
            usercode=usercode,
            role="assistant",
            message=transcribed_text,
            response=json.dumps(reply),
            audio_blob=audio_bytes,
        )

        return jsonify({"voice_text": transcribed_text, "response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/tts")
def tts():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text")

        if not text:
            return jsonify({"error": "text is required"}), 400

        response = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice="alloy", input=text
        )

        audio_bytes = response.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return jsonify(
            {"audio": f"data:audio/mp3;base64,{audio_base64}", "format": "mp3"}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/")
def home():
    return {"message": "Fast SQL Chat with Chart Support Running"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
