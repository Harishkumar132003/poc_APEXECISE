from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import os

load_dotenv()

# =====================================================================
# 1. INITIALIZE DATABASE + CACHE SCHEMA
# =====================================================================

def init_database():
    db_uri = (
        "mysql+mysqlconnector://root:your_new_password@localhost:3306/poc"
    )

    engine_args = {
        "pool_pre_ping": True,
        "pool_size": 20,
        "max_overflow": 40
    }

    return SQLDatabase.from_uri(db_uri, engine_args=engine_args)

db = init_database()

# Cache schema one time
SCHEMA = db.get_table_info()

# =====================================================================
# 2. SYSTEM PROMPTS (SESSION CONTEXT)
# =====================================================================

SYSTEM_SQL_ANALYST = f"""
You are an expert MySQL analyst.
You know the following database schema:

<SCHEMA>
{SCHEMA}
</SCHEMA>

Rules:
- Always produce ONLY SQL.
- Never add explanations.
- Never repeat schema.
- Never ask for schema again.
- Use correct table/column names.
"""

SYSTEM_DATA_ANALYST = """
You are a senior data analyst.
Your job is to interpret SQL results and give clear business-style answers.
Keep answers short, factual, and direct.
Do NOT mention schema or SQL rules in the answer.
"""

# =====================================================================
# 3. GLOBAL LLM CLIENTS
# =====================================================================

# Fast model for SQL generation
llm_sql = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Higher quality model for final answer
llm_answer = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# =====================================================================
# 4. CLEAN SQL
# =====================================================================

def clean_sql(q):
    return (
        q.replace("```sql", "")
         .replace("```", "")
         .strip()
    )

# =====================================================================
# 5. SQL GENERATION (NO SCHEMA SENT EACH TIME)
# =====================================================================

def generate_sql(user_question: str):
    messages = [
        {"role": "system", "content": SYSTEM_SQL_ANALYST},
        {"role": "user", "content": user_question}
    ]

    sql = llm_sql.invoke(messages).content
    return clean_sql(sql)

# =====================================================================
# 6. FINAL ANSWER GENERATION
# =====================================================================

def generate_final_answer(question, sql_query, sql_results, chat_history):
    messages = [
        {"role": "system", "content": SYSTEM_DATA_ANALYST},
        *chat_history,
        {"role": "user", "content": f"""
User Question: {question}
SQL Query: {sql_query}
SQL Result: {sql_results}
"""}
    ]

    return llm_answer.invoke(messages).content

# =====================================================================
# 7. MAIN PIPELINE
# =====================================================================

def process_query(user_query, chat_history):
    # ---- Step 1: Generate SQL ----
    sql = generate_sql(user_query)

    # ---- Step 2: Execute SQL ----
    try:
        sql_result = db.run(sql)
    except Exception as e:
        return f"SQL Error: {e}\nGenerated SQL: {sql}"

    # ---- Step 3: Final Answer ----
    history_slice = chat_history[-2:]

    answer = generate_final_answer(
        user_query,
        sql,
        sql_result,
        history_slice
    )

    return answer

# =====================================================================
# 8. FLASK API
# =====================================================================

app = Flask(__name__)

chat_history = [
    AIMessage(content="Hello! I'm your SQL assistant. Ask me anything.")
]

@app.route("/analyze", methods=["POST"])
def analyze():
    print("Received request:")
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "query field is required"}), 400

    chat_history.append(HumanMessage(content=user_query))
    print("Updated chat history:")

    try:
        response = process_query(user_query, chat_history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    print("Generated response:")

    chat_history.append(AIMessage(content=response))

    return jsonify({
        "query": user_query,
        "response": response
    })

@app.route("/", methods=["GET"])
def home():
    return {"message": "Fast SQL Chat API (session schema) is running"}

# =====================================================================
# 9. RUN SERVER
# =====================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
