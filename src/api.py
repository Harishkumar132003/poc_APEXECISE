from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import os
from sqlalchemy import create_engine, text

load_dotenv()


raw_engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:3306/{os.getenv('DB_NAME')}",
    pool_pre_ping=True
)

def save_chat(usercode, role, message, response=None):
    try:
        query = text("""
            INSERT INTO chat_history (usercode, role, message, response)
            VALUES (:usercode, :role, :message, :response)
        """)
        with raw_engine.begin() as conn:
            conn.execute(query, {
                "usercode": usercode,
                "role": role,
                "message": message,
                "response": response
            })
    except Exception as e:
        print("Chat save error:", e)

def get_chat_by_usercode(usercode):
    query = text("""
        SELECT role, message, response, created_at
        FROM chat_history
        WHERE usercode = :usercode
        ORDER BY id ASC
    """)
    with raw_engine.begin() as conn:
        rows = conn.execute(query, {"usercode": usercode}).fetchall()
    return [dict(r._mapping) for r in rows]




def init_database():
    db_uri = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:3306/{os.getenv('DB_NAME')}"


    engine_args = {
        "pool_pre_ping": True,
        "pool_size": 20,
        "max_overflow": 40
    }

    return SQLDatabase.from_uri(db_uri, engine_args=engine_args)

db = init_database()

# Cache schema
SCHEMA = db.get_table_info()

#print("Database schema cached.",SCHEMA)


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

   Do NOT use poc_distillery.
   Do NOT use poc_retail.
   Do NOT use any other table for dispatch.
   ALWAYS answer from depot perspective using poc_wholesale only.

FOR DISTILLERY USERS (USERCODE does NOT start with 'DEPO'):
   ALWAYS use:
         FROM poc_distillery
         WHERE from_entity_code = '<USERCODE>'
  
     Do NOT use poc_wholesale.
     Do NOT use poc_retail.
     Do NOT use any other table for dispatch.
     ALWAYS answer from depot perspective using poc_wholesale only.
  
SQL RULES:
- Output ONLY raw SQL.
- No markdown.
- No explanation.
- Use exact column names.
- Role = '<ROLE>'.
"""




SYSTEM_DATA_ANALYST = """
You are a senior data analyst.
Rules:
- Interpret the SQL result into a short, clear answer.
- If SQLResult contains NOT_ALLOWED, respond:
  "You do not have permission to access this information."
- Never mention SQL.
- Never mention schema.
"""


# =====================================================================
# 3. LLM CLIENTS
# =====================================================================

llm_sql = ChatOpenAI(model="gpt-4.1", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o", temperature=0)



# =====================================================================
# 4. SQL CLEANER
# =====================================================================

def clean_sql(q):
    return q.replace("```sql", "").replace("```", "").strip()


# =====================================================================
# 5. SQL GENERATION
# =====================================================================

def generate_sql(user_question: str, usercode: str,role: str):
    system_prompt = SYSTEM_SQL_ANALYST \
        .replace("<USERCODE>", usercode) \
        .replace("<ROLE>", role)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    sql = llm_sql.invoke(messages).content
    return clean_sql(sql)


# =====================================================================
# 6. FINAL ANSWER GENERATION
# =====================================================================

def generate_final_answer(question, sql_query, sql_results, chat_history):

    # Handle permission block
    if isinstance(sql_results, list) and len(sql_results) > 0:
        row = sql_results[0]
        if isinstance(row, dict) and row.get("message") == "NOT_ALLOWED":
            return "You do not have permission to access this information."

    print("SQL Results:", sql_query)
    messages = [
        {"role": "system", "content": SYSTEM_DATA_ANALYST},
        *chat_history,
        {
            "role": "user",
            "content": f"""
User Question: {question}
SQL Query: {sql_query}
SQL Result: {sql_results}
"""
        },
    ]

    return llm_answer.invoke(messages).content


# =====================================================================
# 7. MAIN PIPELINE
# =====================================================================

def process_query(user_query, usercode, role, chat_history):

    sql = generate_sql(user_query, usercode, role)

    try:
        sql_result = db.run(sql)
    except Exception as e:
        return f"SQL Error: {e}\nGenerated SQL: {sql}"

    history_slice = chat_history[-2:]

    return generate_final_answer(
        user_query, sql, sql_result, history_slice
    )


# =====================================================================
# 8. FLASK API
# =====================================================================

app = Flask(__name__)

chat_history = [
    AIMessage(content="Hello! I'm your SQL assistant. Ask me anything.")
]

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
        response = process_query(user_query, usercode, role, chat_history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    chat_history.append(AIMessage(content=response))
    save_chat(usercode, "assistant", message=user_query, response=response)

    return jsonify({
        "query": user_query,
        "usercode": usercode,
        "response": response
    })



@app.get("/analyze/history/<usercode>")
def analyze_history(usercode):
    results = get_chat_by_usercode(usercode)
    return jsonify({
        "usercode": usercode,
        "history": results
    })




@app.get("/")
def home():
    return {"message": "Fast SQL Chat API (Depot Optional) is running"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
