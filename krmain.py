import os
import json
import shutil
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pypdf import PdfReader
from hdbcli import dbapi
 
import joblib
from groq import Groq
from dotenv import load_dotenv
 
load_dotenv()  # Load environment variables from .env file
 
# Import an ultra-lightweight open-source vectorization tool
from sklearn.feature_extraction.text import TfidfVectorizer
 
app = FastAPI(title="Lightweight SAP BTP RAG Vector Engine")
 
# Securely initialize the Groq client
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("CRITICAL: GROQ_API_KEY is missing from your .env configuration file!")
 
groq_client = Groq(api_key=GROQ_KEY)
 
# Helper utility to load your shared vectorizer state safely
def load_ingested_vectorizer():
    pkl_filename = "tfidf_vectorizer.pkl"
    if not os.path.exists(pkl_filename):
        raise HTTPException(
            status_code=500,
            detail="Vectorizer model file not found on disk. Please run the data ingestion pipeline first."
        )
    return joblib.load(pkl_filename)
 
# Define vector dimensions footprint.
# 512 dimensions is highly accurate for enterprise documents and extremely light on memory.
VECTOR_DIMENSIONS = 512
 
# --- 1. CONFIGURATION & SAP BTP CONNECTIONS ---
def get_hana_connection():
    """Extracts VCAP_SERVICES credentials to connect to SAP HANA Cloud."""
    vcap_services = os.environ.get("VCAP_SERVICES")
   
    if vcap_services:
        # Running in production on SAP BTP
        services = json.loads(vcap_services)
        hana_config = services["hana"]["credentials"]
       
        # 1. Connect using ONLY standard parameters accepted by hdbcli
        conn = dbapi.connect(
            address=hana_config["host"],
            port=int(hana_config["port"]),
            user=hana_config["user"],
            password=hana_config["password"]
        )
       
        # 2. Dynamically set the schema context right after connecting
        schema_name = hana_config.get("schema", "SYSTEM")
        cursor = conn.cursor()
        cursor.execute(f"SET SCHEMA {schema_name}")
        cursor.close()
       
        return conn
    else:
        # Local fallback testing credentials
        conn = dbapi.connect(
            address="b23d86ce-c107-441b-ac69-e644ddef28e6.hna1.prod-us10.hanacloud.ondemand.com",
            port=443,
            user="DBADMIN",
            password="Shriganesha@24"
        )
       
        # Explicitly set your local workspace schema
        cursor = conn.cursor()
        cursor.execute("SET SCHEMA DBADMIN")
        cursor.close()
       
        return conn
 
# --- 2. TEXT CHUNKING UTILITY ---
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks to preserve semantic context for RAG."""
    words = text.split()
    chunks = []
   
    words_per_chunk = chunk_size // 5  
    words_overlap = chunk_overlap // 5
   
    i = 0
    while i < len(words):
        chunk_words = words[i:i + words_per_chunk]
        chunks.append(" ".join(chunk_words))
        i += (words_per_chunk - words_overlap)
       
    return [c for c in chunks if c.strip()]
 
 
# --- 3. CORE PROCESSING & LIGHTWEIGHT INGESTION ---
def process_pdf_to_vector_db(file_path: str, filename: str):
    """
    Background task to extract, chunk, embed, and store data in SAP HANA.
    Uses dynamic getattr lookups to completely prevent compilation and attribute errors.
    """
    conn = None
    try:
        # --- STEP A: PARSE TEXT VIA PYPDF ---
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + " "
 
        if not full_text.strip():
            print(f"[ERROR] No text extracted from {filename}")
            return
 
        # --- STEP B: TEXT CHUNKING ---
        chunks = chunk_text(full_text)
        if not chunks:
            print(f"[ERROR] Chunking yielded no valid text blocks for {filename}")
            return
 
        # --- STEP C: GENERATE EMBEDDINGS WITH COMPILER-SAFE CONVERSION ---
        vectorizer = TfidfVectorizer(max_features=VECTOR_DIMENSIONS)
        tfidf_matrix = vectorizer.fit_transform(chunks)
 
        # 1. Dynamically resolve conversion methods using string lookups.
        # This completely stops the environment from throwing "Unknown Attribute" errors!
        dense_method = getattr(tfidf_matrix, "toarray", None) or getattr(tfidf_matrix, "todense", None) or getattr(tfidf_matrix, "to_array", None)
 
        if dense_method is not None:
            # Execute the method safely at runtime
            dense_data = dense_method()
           
            # If the output is a legacy NumPy matrix wrapper, convert it to a standard array layout
            if type(dense_data).__name__ == "matrix" or not hasattr(dense_data, "tolist"):
                embeddings = np.asarray(dense_data).tolist()
            else:
                embeddings = dense_data.tolist()
        else:
            # 2. BULLETPROOF FALLBACK: If your environment blocks all sparse conversion methods,
            # force NumPy to unpack the matrix block row-by-row using native array iterations.
            embeddings = [np.asarray(row).ravel().tolist() for row in tfidf_matrix]
 
        # ---> ADD THIS LINE INSIDE YOUR INGESTION CODE BEFORE INSERTING INTO HANA <---
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        print("[SUCCESS] Saved the fitted vectorizer state to tfidf_vectorizer.pkl")
 
 
        # --- STEP D: INSERT INTO SAP HANA CLOUD VECTOR ENGINE ---
        conn = get_hana_connection()
        cursor = conn.cursor()
 
        try:
        # Ensure target table matches the open-source dimensions footprint
            cursor.execute(f"""
                CREATE TABLE ZKPRAG_DOCUMENTS (
                    ID VARCHAR(36) PRIMARY KEY,
                    FILENAME VARCHAR(255),
                    CHUNK_INDEX INT,
                    TEXT_CONTENT NCLOB,
                    VECTOR REAL_VECTOR({VECTOR_DIMENSIONS})
                )
            """)
        except Exception as ea:
            print(f"[INFO] Table creation skipped: {str(ea)}")
            # Error code 288 indicates the table already exists in SAP HANA
            print("Table ZKPRAG_DOCUMENTS already exists. Skipping creation.")
 
        # Prepare batch insert statement
        insert_query = """
            INSERT INTO ZKPRAG_DOCUMENTS (ID, FILENAME, CHUNK_INDEX, TEXT_CONTENT, VECTOR)
            VALUES (?, ?, ?, ?, TO_REAL_VECTOR(?))
        """
        batch_data = []
       
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            doc_id = str(uuid.uuid4())
            # Convert float list to exact string format expected by HANA: '[0.12, 0.0, ...]'
            vector_str = str(vector)
            batch_data.append((doc_id, filename, idx, chunk, vector_str))
 
        # Execute data ingestion batch safely
        cursor.executemany(insert_query, batch_data)
        conn.commit()
        cursor.close()
       
        print(f"[SUCCESS] Ingested {len(batch_data)} chunks into SAP HANA for file: {filename}")
 
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to process vector insertion: {str(e)}")
       
    finally:
        # Clean up database resource links
        if conn:
            conn.close()
        # Clean up temporary disk space in the cloud container
        if os.path.exists(file_path):
            os.remove(file_path)
 
 
# --- 4. API ENDPOINT FOR FRONTEND CONSUMPTION ---
@app.post("/api/v1/ingest")
async def upload_and_vectorize(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Main endpoint for your Fiori UI.
    Accepts Multipart Form Data file upload, saves temporarily, and triggers background processing.
    """
    # FIX: Safely cast the filename to a string first to handle FastAPI wrapper structures
    filename_str = str(file.filename)
   
    if not filename_str.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
 
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
   
    # Use the safe string variable for the file path construction
    temp_file_path = os.path.join(temp_dir, filename_str)
 
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
           
        # Pass the verified string filename to your background thread execution task
        background_tasks.add_task(process_pdf_to_vector_db, temp_file_path, filename_str)
 
        return {
            "status": "processing",
            "message": f"File '{filename_str}' uploaded successfully. Vector generation is running in the background.",
            "target_endpoint": "/api/v1/ingest"
        }
       
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Upload initialization failed: {str(e)}")
 
 
# --- 5. CHAT QUERY ENDPOINT (Matches your Fiori controller 'fetch' method) ---
@app.post("/api/v1/query")
async def query_rag_workspace(payload: dict):
    """
    Receives prompt text from the Fiori View, vectorizes it matching ingestion dimensions,
    queries SAP HANA Cloud via vector similarity, and streams the context to Groq.
    """
    user_question = payload.get("question", "")
    if not user_question:
        raise HTTPException(status_code=400, detail="Empty text string received.")
   
    conn = None
    try:
        # --- STEP A: VECTORIZE USER QUESTION WITH COMPILER-SAFE LAYOUT ---
        # Retrieve the vectorizer that holds your dictionary vocabulary mappings
        vectorizer = load_ingested_vectorizer()
       
        # Transform the single query string into a sparse matrix footprint
        query_matrix = vectorizer.transform([user_question])
 
        # Exact dynamic safe lookups matching your specific ingestion fallback sequence
        dense_method = getattr(query_matrix, "toarray", None) or getattr(query_matrix, "todense", None) or getattr(query_matrix, "to_array", None)
 
        if dense_method is not None:
            dense_data = dense_method()
            # If the output layout is a legacy NumPy matrix wrapper, flatten to basic array
            if type(dense_data).__name__ == "matrix" or not hasattr(dense_data, "tolist"):
                query_vector = np.asarray(dense_data).ravel().tolist()
            else:
                query_vector = dense_data[0].tolist() if isinstance(dense_data, np.ndarray) else dense_data.tolist()
        else:
            # Bulletproof fallback unpack execution block
            query_vector = np.asarray(query_matrix.todense()).ravel().tolist()
 
        # Convert float list to the exact string array layout format expected by TO_REAL_VECTOR: '[0.1, 0.0, ...]'
        query_vector_str = str(query_vector)
 
        # --- STEP B: EXECUTE COSINE SIMILARITY MATCH AGAINST SAP HANA CLUID ---
        # (Assuming your custom get_hana_connection function is imported/declared globally)
        conn = get_hana_connection()
        cursor = conn.cursor()
 
        # Query top 3 closest related segments based on database vector spatial geometry
        search_query = """
            SELECT TOP 3 TEXT_CONTENT
            FROM ZKPRAG_DOCUMENTS
            ORDER BY COSINE_SIMILARITY(VECTOR, TO_REAL_VECTOR(?)) DESC
        """
        cursor.execute(search_query, (query_vector_str,))
        rows = cursor.fetchall()
        cursor.close()
 
        # Extract text elements out from the database result matrix tuples
        context_chunks = [row[0] for row in rows if row and row[0]]
        context_text = "\n\n---\n\n".join(context_chunks)
 
        if not context_text:
            context_text = "No related document background records were found matching your query."
 
        # --- STEP C: BUILD SYSTEM AND USER COMPILER INSTRUCTIONS ---
        system_prompt = (
            "You are an enterprise AI assistant integrated with an SAP Fiori RAG architecture. "
            "Your task is to answer the user's question accurately using ONLY the provided document context fragments. "
            "If the context does not contain the information needed to formulate an answer, "
            "say: 'I cannot find the answer in the uploaded enterprise documentation.'"
        )
 
        user_prompt = f"""
Context from uploaded enterprise documents:
==================================================
{context_text}
==================================================
 
User Question: {user_question}
 
Answer:
"""
 
        # --- STEP D: GENERATE CONSTRAINED ANSWER USING GROQ API ---
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,  # 0.0 forces strict adherence to text facts, preventing hallucinations
            max_tokens=1024
        )
 
        ai_response = chat_completion.choices[0].message.content
 
        # --- STEP E: RESPOND BACK TO SAP FIORI CONTROLLER PAYLOAD ---
        return {
            "status": "success",
            "answer": ai_response
        }
 
    except Exception as e:
        print(f"[CRITICAL ERROR IN QUERY PIPELINE]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal application vector matching processing failure: {str(e)}"
        )
 
    finally:
        # Keep connection pathways clean and unlinked
        if conn:
            conn.close()
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
 
 