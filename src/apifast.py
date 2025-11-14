from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import os
import traceback

load_dotenv()
app = Flask(__name__)

# =====================================================================
# 1. INIT DB + CACHE SCHEMA
# =====================================================================
def init_database():
    db_uri = os.environ.get(
        "CLASSICMODELS_DB_URI",
        "mysql+mysqlconnector://root:your_new_password@localhost:3306/classicmodels"
    )
    engine_args = {"pool_pre_ping": True, "pool_size": 20, "max_overflow": 40}
    return SQLDatabase.from_uri(db_uri, engine_args=engine_args)

db = init_database()
SCHEMA = db.get_table_info()  # cached at startup

# =====================================================================
# 2. PROMPTS / SYSTEM MESSAGES
# =====================================================================
SYSTEM_SQL = f"""
You are an expert MySQL analyst.
Use this schema:

<schema>
{SCHEMA}
</schema>

Rules:
- Return ONLY SQL.
- No explanation.
- No markdown/backticks.
- Use correct table and column names.
"""

SYSTEM_ANSWER = """
You are a senior data analyst who writes short, direct, business-style answers.
Do NOT include SQL or schema in the final answer.
Keep it concise and factual.
"""

# =====================================================================
# 3. MODELS
# =====================================================================
# fast model to produce SQL
llm_sql = ChatOpenAI(model=os.environ.get("LLM_SQL_MODEL", "gpt-4o-mini"), temperature=0)

# higher-quality model to generate final answer
llm_answer = ChatOpenAI(model=os.environ.get("LLM_ANSWER_MODEL", "gpt-4o"), temperature=0)

# =====================================================================
# 4. PROMPT TEMPLATES + RUNNABLE CHAINS
# =====================================================================
sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_SQL),
    ("user", "{question}")
])

# sql_chain returns plain SQL string via StrOutputParser
sql_chain = sql_prompt | llm_sql | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_ANSWER),
    ("user", """
User Question: {question}
SQL Query: {sql}
SQL Result: {result}
""")
])

# answer_chain returns plain text final answer
answer_chain = answer_prompt | llm_answer | StrOutputParser()

# =====================================================================
# 5. SQL execution helper
# =====================================================================
def clean_sql(q: str) -> str:
    if not isinstance(q, str):
        return str(q)
    return q.replace("```sql", "").replace("```", "").strip()

def execute_sql(sql: str):
    sql = clean_sql(sql)
    try:
        res = db.run(sql)
    except Exception as e:
        # Attach stacktrace for debugging (but return readable error)
        tb = traceback.format_exc()
        return {"error": "SQL Execution Error", "message": str(e), "trace": tb}
    # Try to make result JSON-serializable
    try:
        # If the result is already a list of dicts or similar, return it
        return res
    except Exception:
        return str(res)

# =====================================================================
# 6. COMPOSED RUNNABLE PIPELINE
# =====================================================================
# Step A: produce SQL from question
# Step B: execute SQL
# Step C: produce final answer
full_pipeline = (
    RunnableMap({
        "question": RunnablePassthrough(),   # pass original question through
        "sql": sql_chain                     # generate SQL string
    })
    | RunnableMap({
        # build context for the final answer
        "question": lambda ctx: ctx["question"],
        "sql": lambda ctx: clean_sql(ctx["sql"]),
        "result": lambda ctx: execute_sql(clean_sql(ctx["sql"]))
    })
    | answer_chain  # produce final human answer string
)

# =====================================================================
# 7. FLASK ROUTE (non-streaming JSON)
# =====================================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    question = payload.get("query") or payload.get("question") or payload.get("q")
    if not question:
        return jsonify({"error": "query field is required"}), 400

    try:
        # Run the pipeline; this will:
        # 1) generate SQL (fast model)
        # 2) execute SQL
        # 3) generate final answer (quality model)
        final_answer = full_pipeline.invoke(question)

        # Also generate SQL separately (useful to return for debugging)
        generated_sql = (sql_chain.invoke(question))
        sql_result = execute_sql(clean_sql(generated_sql))

        return jsonify({
            "query": question,
            "sql": clean_sql(generated_sql),
            "sql_result": sql_result,
            "answer": final_answer
        })
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "internal_error", "message": str(e), "trace": tb}), 500

# =====================================================================
# 8. ROOT
# =====================================================================
@app.route("/", methods=["GET"])
def home():
    return {"message": "Fast SQL Chat API (normal JSON) is running"}

# =====================================================================
# 9. RUN SERVER
# =====================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
