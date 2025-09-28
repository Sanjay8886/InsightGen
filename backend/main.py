from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import io
import json
import requests
from datetime import datetime
import threading
import os
import sys
import traceback
from contextlib import contextmanager
import re
import numpy as np
import logging

# --- Basic logging for debugging ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("insightgen")

# --- Configuration ---
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://insightgen-production.up.railway.app"  # <--- THIS IS THE KEY ADDITION
        ],
        "supports_credentials": True,  # Recommended if you use cookies or session headers
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"] # Recommended to be specific
    }
})
DATABASE_NAME = 'insightgen_log.db'
LOCK = threading.Lock()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
ACTIVE_DATA_TABLE = 'uploaded_data'

CONVERSATION_HISTORY = []
MAX_HISTORY_LENGTH = 10

if not GEMINI_API_KEY:
    logger.error("FATAL ERROR: GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable.")

# --- DB helpers ---
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except Exception as e:
        logger.exception("Database connection error:")
        raise
    finally:
        if conn:
            conn.close()

def initialize_database():
    logger.debug("Initializing database tables...")
    try:
        with get_db_connection() as db_connection:
            cursor = db_connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    file_name TEXT,
                    user_query TEXT,
                    generated_sql TEXT,
                    result_summary TEXT
                )
            """)
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {ACTIVE_DATA_TABLE} (placeholder TEXT)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS upload_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    file_name TEXT,
                    row_count INTEGER,
                    column_count INTEGER,
                    quality_report TEXT
                )
            """)
            logger.debug("DB initialization complete.")
    except Exception as e:
        logger.exception("Error during database initialization:")

initialize_database()

# ----------------------------
# Data cleaning & utilities
# ----------------------------
def _normalize_col_name(col: str, idx: int = None) -> str:
    if col is None:
        col = ""
    col = str(col).strip()
    # Normalize: replace non-alphanumeric/underscore with underscore, strip leading/trailing underscores, force lowercase
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', col).strip('_').lower()
    if cleaned == '':
        cleaned = f"col_{idx}" if idx is not None else f"col_{abs(hash(col)) % (10**6)}"
    # Prepend 'c_' if it starts with a digit
    if re.match(r'^\d', cleaned):
        cleaned = f"c_{cleaned}"
    return cleaned

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = list(df.columns)
    new_cols = []
    for i, c in enumerate(original_cols):
        new_cols.append(_normalize_col_name(c, i))
    df.columns = new_cols

    before_cols = df.shape[1]
    df = df.dropna(axis=1, how='all')
    after_cols = df.shape[1]
    logger.debug(f"Dropped {before_cols - after_cols} fully-empty columns.")

    before_rows = df.shape[0]
    df = df.dropna(axis=0, how='all')
    after_rows = df.shape[0]
    logger.debug(f"Dropped {before_rows - after_rows} fully-empty rows.")

    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df.loc[df[col] == '', col] = pd.NA
        except Exception:
            continue

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(50).tolist()
        numeric_like = 0
        for s in sample:
            try:
                float(str(s).replace(',', ''))
                numeric_like += 1
            except Exception:
                pass
        if len(sample) > 0 and numeric_like / len(sample) > 0.75:
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                logger.debug(f"Column {col} coerced to numeric.")
            except Exception:
                pass

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(50).tolist()
        date_like = 0
        for s in sample:
            try:
                pd.to_datetime(s, infer_datetime_format=True)
                date_like += 1
            except Exception:
                pass
        if len(sample) > 0 and date_like / len(sample) > 0.6:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                logger.debug(f"Column {col} coerced to datetime.")
            except Exception:
                pass

    return df

def create_quality_report(df: pd.DataFrame) -> dict:
    report = {}
    report['rows'] = int(df.shape[0])
    report['columns'] = int(df.shape[1])
    missing = (df.isna().sum() / max(1, df.shape[0])) * 100
    missing = missing.sort_values(ascending=False)
    # convert numpy floats to native floats for JSON safety
    report['top_missing_percent'] = {k: float(v) for k, v in missing.head(10).to_dict().items()}
    single_val_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    report['single_value_columns'] = single_val_cols
    sample_preview = {}
    for col in df.columns[:5]:
        sample_preview[col] = df[col].dropna().astype(str).head(3).tolist()
    report['preview'] = sample_preview
    return report

# ----------------------------
# Persistence helpers
# ----------------------------
def load_df_from_db():
    """Loads the currently active data from the SQLite database."""
    try:
        with get_db_connection() as conn:
            # We use an explicit table name, which is safe from user input
            df = pd.read_sql(f"SELECT * FROM {ACTIVE_DATA_TABLE}", conn)
            if df.empty:
                logger.debug(f"Table {ACTIVE_DATA_TABLE} exists but returned empty DataFrame.")
                if 'placeholder' in df.columns and len(df.columns) == 1:
                    logger.debug("Found only placeholder column.")
                return None, None
            # Drop the placeholder column if it exists alongside real data
            if 'placeholder' in df.columns and len(df.columns) > 1:
                df = df.drop(columns=['placeholder'], errors='ignore')
            logger.debug(f"Loaded {len(df)} rows / {len(df.columns)} cols from persistent DB.")
            return df, "uploaded_file"
    except Exception as e:
        # This catches errors like the table not existing or being corrupt
        logger.exception("Persistence load failed: Error reading from database.")
        return None, None

# ----------------------------
# LLM helper (robust errors)
# ----------------------------
def call_gemini_api(system_instruction, contents_array):
    """
    Calls the Gemini API with exponential backoff and robust error handling.
    """
    cleaned_api_key = GEMINI_API_KEY.strip().replace('$', '') if GEMINI_API_KEY else ""
    if not cleaned_api_key:
        logger.debug("API key is empty in call_gemini_api.")
        return "Error: AI API Key not found or is empty."

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": contents_array,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    url = GEMINI_API_URL + "?key=" + cleaned_api_key
    max_retries = 3
    delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            return text
        except requests.exceptions.HTTPError as http_err:
            try:
                status = response.status_code
            except Exception:
                status = None
            if status == 403:
                return "Error: 403 Forbidden. Please check your GEMINI_API_KEY."
            if status == 400:
                # Log the body if it's a 400 error to help debug payload issues
                logger.error(f"400 Bad Request Payload: {json.dumps(payload)}")
                return "Error: 400 Bad Request. Possibly malformed key or payload."
            if attempt < max_retries - 1 and (status == 429 or status >= 500):
                import time
                time.sleep(delay)
                delay *= 2
            else:
                logger.exception("Gemini API request failed:")
                return f"Error: Failed to connect to AI service. Detail: {http_err}"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(delay)
                delay *= 2
            else:
                logger.exception("Gemini API connection exception:")
                return f"Error: AI service connection failed. Detail: {e}"

    return "Error: AI service connection failed after multiple retries."

# ----------------------------
# Deterministic file description endpoint (NO LLM) 
# ----------------------------
@app.route('/describe', methods=['GET'])
def describe_file():
    """
    Deterministic summary of the currently uploaded file WITHOUT calling the LLM.
    This answers questions like "what is the file about?" even when LLM is down.
    """
    df.file_name = load_df_from_db()
    if df is None:
        return jsonify({"status": "error", "error": "No uploaded data found. Upload a CSV or Excel file first."}), 404

    try:
        # Ensure dtypes are string for safe JSON serialization
        schema = [{ "column": col, "dtype": str(dtype) } for col, dtype in df.dtypes.items()]
        row_count, col_count = df.shape
        # Top-3 frequent values for first 5 columns (if available)
        frequent = {}
        for col in df.columns[:5]:
            try:
                top = df[col].dropna().astype(str).value_counts().head(3).to_dict()
                frequent[col] = top
            except Exception:
                frequent[col] = []
        preview = df.head(5).replace({np.nan: None}).to_dict(orient='records')
        quality_report = create_quality_report(df)

        # A friendly assistant-style summary message
        assistant_summary = (
            f"I have loaded **'{file_name or 'uploaded_file'}'** with **{row_count} rows** and **{col_count} columns**. "
            f"The data is ready for analysis."
        )

        return jsonify({
            "status": "success",
            "summary": assistant_summary,
            "schema": schema,
            "preview": preview,
            "frequent_values_sample": frequent,
            "quality_report": quality_report
        }), 200
    except Exception as e:
        logger.exception("Error building file description:")
        return jsonify({"status": "error", "error": f"Failed to describe file: {e}"}), 500

# ----------------------------
# Upload route (robust)
# ----------------------------
@app.route('/upload', methods=['POST'])
def upload():
    logger.debug("Received upload request.")
    global CONVERSATION_HISTORY
    # Reset conversation history on new upload
    CONVERSATION_HISTORY = []

    if 'file' not in request.files:
        return jsonify({"status": "error", "error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "error": "No selected file"}), 400

    original_filename = file.filename
    ext = original_filename.split('.')[-1].lower()
    if ext not in ['csv', 'xlsx', 'xls']:
        return jsonify({"status": "error", "error": f"Invalid file type: .{ext}. Only CSV and Excel files are supported."}), 400

    try:
        # Read file content safely
        if ext == 'csv':
            content = file.stream.read().decode("utf-8", errors='replace')
            stream = io.StringIO(content)
            df = pd.read_csv(stream, dtype=object)
        else:
            stream = io.BytesIO(file.stream.read())
            # Try to infer the best sheet in Excel, falling back to the default read
            try:
                xls = pd.read_excel(stream, sheet_name=None, engine='openpyxl')
                best_sheet_name, best_df = None, None
                for name, sheet_df in xls.items():
                    if best_df is None or sheet_df.shape[0] > best_df.shape[0]:
                        best_sheet_name, best_df = name, sheet_df
                df = best_df if best_df is not None else pd.DataFrame()
                logger.debug(f"Read Excel, selected sheet: {best_sheet_name}")
            except Exception:
                stream.seek(0)
                df = pd.read_excel(stream, engine='openpyxl')

        if df is None or df.empty:
            logger.warning("Uploaded file parsed to empty DataFrame.")
            return jsonify({"status": "error", "error": "Uploaded file contained no readable data."}), 400

        df = clean_dataframe(df)
        quality_report = create_quality_report(df)
        qr_json = json.dumps(quality_report)

        # Persistence: Replace the existing table with new data
        with get_db_connection() as conn:
            # Use lock only for critical write operation
            with LOCK:
                # Use to_sql with if_exists='replace'
                df.to_sql(ACTIVE_DATA_TABLE, conn, index=False, if_exists='replace')
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO upload_metadata (timestamp, file_name, row_count, column_count, quality_report) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), original_filename, df.shape[0], df.shape[1], qr_json)
                )

        schema_info = ", ".join(f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items())
        # Add initial context to conversation history for LLM
        initial_context = (
            f"New data uploaded successfully. The active table is '{ACTIVE_DATA_TABLE}' "
            f"with schema: {schema_info}. Use this table for all subsequent analysis."
        )
        CONVERSATION_HISTORY.append({'role': 'model', 'content': initial_context})

        assistant_message = f"File '{original_filename}' uploaded and processed. It has **{df.shape[0]} rows** and **{df.shape[1]} columns**."
        return jsonify({
            "status": "success",
            "message": assistant_message,
            "file_name": original_filename,
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "quality_report": quality_report
        }), 200

    except Exception as e:
        logger.exception("File processing error:")
        traceback.print_exc()
        # Friendly fallback so frontend doesn't display raw JSON error
        return jsonify({
            "status": "error",
            "error": "Server error during file processing.",
            "detail": str(e)
        }), 500

# ----------------------------
# Query route (LLM-backed, robust checks + friendly errors)
# ----------------------------
@app.route('/query', methods=['POST'])
def query_data():
    logger.debug("Received query request.")
    df, file_name = load_df_from_db()
    if df is None:
        return jsonify({
            "status": "error",
            "error": "No data found for analysis.",
            "summary": "Please upload a data file (CSV or Excel) to begin the analysis."
        }), 404

    user_query = ""
    if request.is_json:
        data = request.get_json()
        user_query = data.get('query', '').strip()
    if not user_query:
        return jsonify({"status": "error", "error": "Missing natural language query."}), 400

    # Get the normalized schema information for the LLM
    schema_info = ", ".join(f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items())
    global CONVERSATION_HISTORY
    # Keep only the last N turns to avoid context overflow
    CONVERSATION_HISTORY = CONVERSATION_HISTORY[-MAX_HISTORY_LENGTH:]

    # Prepare conversation contents for the API call
    full_contents = []
    for turn in CONVERSATION_HISTORY:
        full_contents.append({"role": turn['role'], 'parts': [{"text": turn['content']}]})

    # --- CLASSIFICATION: YES / NO / NEEDS_CLARIFICATION ---
    classification_system_instruction = (
        f"You are a short classifier. Data schema: {schema_info}. Based on conversation history, determine whether the latest user message requires "
        "a SQL database query (output EXACTLY YES), is purely conversational (output EXACTLY NO), or is ambiguous/vague and needs a targeted clarifying question (output EXACTLY NEEDS_CLARIFICATION)."
    )

    classification_contents = full_contents + [{
        "role": "user",
        "parts": [{"text": f"User's latest message: '{user_query}'. Does this require a data query? Answer YES or NO or NEEDS_CLARIFICATION."}]
    }]

    classification_result_raw = call_gemini_api(classification_system_instruction, classification_contents)
    classification_result = classification_result_raw.upper().strip() if isinstance(classification_result_raw, str) else ""
    logger.debug(f"Classification raw: {classification_result_raw}")

    # Detect AI errors robustly
    if isinstance(classification_result_raw, str) and classification_result_raw.startswith("Error:"):
        logger.error("LLM classification error: " + classification_result_raw)
        # fallback: ask the deterministic describe endpoint if user asks about file
        if re.search(r'\b(what|which|about|describe|columns)\b', user_query, re.IGNORECASE):
            return describe_file()
        return jsonify({"status": "error", "error": "AI service unavailable. Try again later."}), 500

    # --- CLARIFICATION required ---
    if classification_result == 'NEEDS_CLARIFICATION':
        clarify_instruction = (
            "You are a helpful assistant. The user's query is ambiguous for the current dataset. "
            "Provide a single, specific, concise clarifying question (one sentence) that, when answered, will allow generating the correct SQL query. "
            "Do NOT include additional text."
        )
        clarify_contents = full_contents + [{"role": "user", "parts": [{"text": user_query}]}]
        clarifying_question_raw = call_gemini_api(clarify_instruction, clarify_contents)
        if isinstance(clarifying_question_raw, str) and clarifying_question_raw.startswith("Error:"):
            # fallback deterministic approach
            clarifying_question = "Could you clarify exactly which columns or filters you want applied to the dataset?"
        else:
            clarifying_question = clarifying_question_raw.strip()
        
        # Update history
        CONVERSATION_HISTORY.append({'role': 'user', 'content': user_query})
        CONVERSATION_HISTORY.append({'role': 'model', 'content': clarifying_question})

        return jsonify({"status": "clarify", "clarifying_question": clarifying_question}), 200

    # --- NO Query needed (Conversational) ---
    if classification_result != 'YES':
        conversational_system_instruction = (
            "You are a friendly concise conversational AI data agent. Respond in a single short sentence and engage with conversations."
        )
        conversational_summary_raw = call_gemini_api(conversational_system_instruction, full_contents + [{"role": "user", "parts": [{"text": user_query}]}])
        if isinstance(conversational_summary_raw, str) and conversational_summary_raw.startswith("Error:"):
            conversational_summary = "I am ready to analyze your uploaded file — ask a data-specific question (e.g., 'show top 5 rows' or 'what columns are present?')."
        else:
            conversational_summary = conversational_summary_raw.strip()
        
        # Update history
        CONVERSATION_HISTORY.append({'role': 'user', 'content': user_query})
        CONVERSATION_HISTORY.append({'role': 'model', 'content': conversational_summary})

        return jsonify({"status": "success", "summary": conversational_summary, "chart_type": "text_only"}), 200

    # ----------- Proceed with SQL generation -----------
    sql_system_instruction = (
        f"You are an expert SQLite SQL generator. Your task is to generate a single, valid SELECT query against the '{ACTIVE_DATA_TABLE}' table based on the user's request and conversation history. "
        "STRICT RULES: 1. Use ONLY the table and column names provided in the schema (which are all lowercase and cleaned). 2. Use standard SQLite syntax. 3. Use single quotes (') for all string literals. 4. Do NOT use any DDL, DML, or subqueries unless strictly necessary for the analysis. "
        "OUTPUT FORMAT: Output ONLY the raw SQL query. If the user requested schema/columns, output EXACTLY the token ONLY_SCHEMA_SUMMARY. If the request is purely conversational or non-data, output EXACTLY ONLY_SUMMARY."
        f"you are smart enough to handle vague questions, if user is confused and gives confused queries its your job to understand them clearly and generate necessar queries."
        f"remeber u should handle dirty and inavlid data in the dtaabase and provide user with necessary answers"
        f" Table Schema (Columns and SQLite types): {schema_info}"
    )
    generated_sql_raw = call_gemini_api(sql_system_instruction, full_contents + [{"role": "user", "parts": [{"text": user_query}]}])
    logger.debug(f"Generated SQL raw: {generated_sql_raw}")

    if isinstance(generated_sql_raw, str) and generated_sql_raw.startswith("Error:"):
        logger.error("LLM SQL generation error: " + generated_sql_raw)
        return jsonify({"status": "error", "error": "AI service failed to generate SQL. Try again or use a simpler question."}), 500

    generated_sql = generated_sql_raw.strip()

    # --- ONLY_SUMMARY or ONLY_SCHEMA_SUMMARY ---
    if generated_sql.upper() in ['ONLY_SUMMARY', 'ONLY_SCHEMA_SUMMARY']:
        if generated_sql.upper() == 'ONLY_SCHEMA_SUMMARY':
            analysis_summary = f"The data is stored in **'{ACTIVE_DATA_TABLE}'**. Columns and types: {schema_info}"
        else:
            summary_system_instruction = ("You are a helpful assistant. Answer the user's high-level question about their data concisely using conversation history.")
            analysis_summary_raw = call_gemini_api(summary_system_instruction, full_contents + [{"role": "user", "parts": [{"text": user_query}]}])
            if isinstance(analysis_summary_raw, str) and analysis_summary_raw.startswith("Error:"):
                analysis_summary = "I can provide a high-level summary, but the AI service is temporarily unavailable."
            else:
                analysis_summary = analysis_summary_raw.strip()
        
        # Update history
        CONVERSATION_HISTORY.append({'role': 'user', 'content': user_query})
        CONVERSATION_HISTORY.append({'role': 'model', 'content': analysis_summary})

        return jsonify({"status": "success", "summary": analysis_summary, "chart_type": "text_only"}), 200

    # --- SQL Cleanup and Validation ---
    select_match = re.search(r"(select\b[\s\S]*?;?$)", generated_sql, re.IGNORECASE)
    if select_match:
        # Extract only the SELECT statement
        generated_sql_clean = select_match.group(1).strip().rstrip(';')
    else:
        generated_sql_clean = generated_sql.strip()

    if not re.match(r'^\s*select\b', generated_sql_clean, re.IGNORECASE):
        logger.warning(f"AI did not produce a valid SELECT statement: {generated_sql_clean}")
        return jsonify({
            "status": "error",
            "error": f"AI failed to generate a valid SQL query. Output: {generated_sql_clean}",
            "summary": "I couldn't translate your question into a valid SELECT query. Try asking for columns, top rows, or simpler filters."
        }), 500

    # Execute SQL (safe path)
    analysis_summary = ""
    table_html = ""
    result_data_json = "[]"
    chart_type = "table"

    try:
        # Execute with connection context manager
        with get_db_connection() as conn:
            sql_result_df = pd.read_sql_query(generated_sql_clean, conn)

        num_columns = len(sql_result_df.columns)
        num_rows = len(sql_result_df)
        logger.debug(f"SQL result -> cols: {num_columns}, rows: {num_rows}")

        if sql_result_df.empty:
            analysis_summary = "The generated query executed successfully but returned **no rows**. Check your filters or try a different question."
            table_html = ""
            result_data_json = "[]"
            chart_type = "table"
        else:
            # AI Summary and Chart Recommendation
            summary_system_instruction = (
                "You are an expert data analyst. Provide a concise summary of the SQL results and recommend a chart type using the tag [CHART_TYPE:TYPE]. "
                "The TYPE should be one of: 'bar', 'line', 'pie', 'scatter', or 'table'."
            )
            summary_prompt = (
                f"Analyze SQL result for user question.\n"
                f"Result has {num_columns} columns and {num_rows} rows.\n"
                f"SQL: {generated_sql_clean}\n"
                f"First 5 rows:\n{sql_result_df.head().to_string()}"
            )
            raw_ai_response = call_gemini_api(summary_system_instruction, full_contents + [{"role": "user", "parts": [{"text": summary_prompt}]}])
            
            if isinstance(raw_ai_response, str) and raw_ai_response.startswith("Error:"):
                # Fallback deterministic summary
                top_preview = sql_result_df.head(3).replace({np.nan: None}).to_dict(orient='records')
                analysis_summary = f"Query returned **{num_rows} rows** and **{num_columns} columns**. Preview (first 3 rows): {top_preview}"
                chart_type = 'table'
            else:
                chart_type_match = re.search(r'\[CHART_TYPE:(\w+)\]', raw_ai_response, re.IGNORECASE)
                chart_type = chart_type_match.group(1).lower() if chart_type_match else 'table'
                analysis_summary = re.sub(r'\[CHART_TYPE:(\w+)\]', '', raw_ai_response).strip()

            # Prepare output data
            should_show_table = num_rows > 0
            if should_show_table:
                # Use Tailwind classes for simple styling
                table_html = sql_result_df.to_html(classes='table-auto w-full rounded-lg shadow-md', index=False, border=0)
            else:
                table_html = ""

            # Prepare data for chart generation (replace NaN/NaT with None for JSON)
            result_data_json = sql_result_df.replace({np.nan: None}).to_json(orient='records')

        # Update history with the executed SQL and result
        CONVERSATION_HISTORY.append({'role': 'user', 'content': user_query})
        model_action = f"[Executed SQL: {generated_sql_clean}] [Summary: {analysis_summary}] [Chart: {chart_type}]"
        CONVERSATION_HISTORY.append({'role': 'model', 'content': model_action})

    except Exception as e:
        logger.exception(f"SQL execution error for query: {generated_sql_clean}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "Database error while running the query.It seems the database contains dirty or invalid data kindly check.",
            "summary": f"I couldn't run the generated SQL query due to a database error: **{str(e)}**."
        }), 500

    # Logging successful request
    try:
        with LOCK:
            with get_db_connection() as db_connection:
                cursor = db_connection.cursor()
                cursor.execute(
                    "INSERT INTO analysis_logs (timestamp, file_name, user_query, generated_sql, result_summary) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), file_name, user_query, generated_sql_clean[:2000], analysis_summary[:500])
                )
    except Exception:
        logger.exception("Error logging request:")

    return jsonify({
        "status": "success",
        "summary": analysis_summary,
        "table_html": table_html,
        "result_data_json": result_data_json,
        "chart_type": chart_type
    }), 200

# --- Run app ---
if __name__ == '__main__':
    logger.info("Starting AI Data Agent Backend...")
    app.run(debug=True, port=5001, use_reloader=False)
