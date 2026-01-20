"""
This script defines a FastAPI server that uses OpenAI's RAG model to generate answers for user queries.
The server retrieves relevant documents using a local retrieval service, reranks the documents using VoyageAI's reranker, 
and then generates an answer using OpenAI's RAG model.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import logging
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import gunicorn.app.base
load_dotenv('.env')
app = FastAPI()

LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("rag_server")

RETRIEVE_URL = os.getenv("RAG_RETRIEVE_URL", "http://127.0.0.1:4004/v1/retrieve")
USER_RETRIEVE_URL = os.getenv("RAG_USER_RETRIEVE_URL", "http://127.0.0.1:4006/v1/retrieve")
RETRIEVE_TIMEOUT = float(os.getenv("RAG_RETRIEVE_TIMEOUT", "10"))
RERANK_TIMEOUT = float(os.getenv("RAG_RERANK_TIMEOUT", "10"))
RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
RERANK_MAX_DOCS = int(os.getenv("RAG_RERANK_MAX_DOCS", "0"))
CONTEXT_MAX_DOCS = int(os.getenv("RAG_CONTEXT_MAX_DOCS", "0"))
RERANK_TEXT_MAX_CHARS = int(os.getenv("RAG_RERANK_TEXT_MAX_CHARS", "0"))
CONTEXT_TEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_TEXT_MAX_CHARS", "0"))
USER_RETRIEVE_MULTIPLIER = int(os.getenv("RAG_RETRIEVE_MULTIPLIER", "6"))
USER_RETRIEVE_MIN_K = int(os.getenv("RAG_RETRIEVE_MIN_USER_K", "20"))
USER_RETRIEVE_MAX_K = int(os.getenv("RAG_RETRIEVE_MAX_USER_K", "120"))

SESSION = requests.Session()


# OpenAI client configuration
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

api_key = os.getenv('OPEN_AI_API_KEY_30')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

def get_client(model: str):
    if "deepseek" in model:
        return OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    return OpenAI(api_key=api_key)

client = get_client("gpt-4o-mini") # Default client


# Add VoyageAI configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

class Query(BaseModel):
    query: str
    source: Optional[str] = " The source is available in the context "
    max_tokens: int = 1000
    num_docs: int = 5
    destination: Optional[str] = None 
    model: Optional[str] = "gpt-4o-mini" 
    user_id: Optional[str] = None
    thread_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str


def _truncate_text(text: str, max_chars: int) -> str:
    if not max_chars or max_chars <= 0:
        return text
    return text[:max_chars]


def _cap_docs(docs: List[Dict[str, Any]], max_docs: int) -> List[Dict[str, Any]]:
    if not max_docs or max_docs <= 0:
        return docs
    return docs[:max_docs]


def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank documents using VoyageAI reranker.
    If the reranker service is not available, the original document order is used.
    """
    if not documents:
        logger.info("No relevant documents retrieved for reranking.")
        return [{"message": "No relevant documents retreived from user input. Ask the user to provide relevant documents"}]
    if not RERANK_ENABLED or not VOYAGE_API_KEY:
        logger.info("Reranking disabled or missing VoyageAI key. Using original document order.")
        return documents
    try:
        # Extract text from documents
        doc_texts = [
            _truncate_text(doc.get("text", ""), RERANK_TEXT_MAX_CHARS)
            for doc in documents
        ]
        
        headers = {
            "Authorization": f"Bearer {VOYAGE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "documents": doc_texts,
            "model": "rerank-2",  # Using their recommended model
            "return_documents": True
        }
        
        response = SESSION.post(
            VOYAGE_RERANK_URL, headers=headers, json=payload, timeout=RERANK_TIMEOUT
        )
        response.raise_for_status()
        reranked_results = response.json()
        
        # Reconstruct documents with reranked order and scores
        reranked_docs = []
        for result in reranked_results["data"]:
            original_doc = documents[result["index"]]
            original_doc["relevance_score"] = result["relevance_score"]
            reranked_docs.append(original_doc)
        return reranked_docs
        
    except Exception as e:
        logger.warning("Reranking failed, using original order: %s", e)
        return documents

def query_retrieval_service(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the local retrieval service for relevant documents.
    """
    try:
        encoded_query = quote(query)
        response = SESSION.get(
            f"{RETRIEVE_URL}?query={encoded_query}&k={k}",
            timeout=RETRIEVE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying retrieval service: {str(e)}")

def query_retrieval_service2(query: str, k: int = 2) -> List[Dict[str, Any]]:
    """
    Query the local retrieval service for relevant documents
    ."""
    try:
        encoded_query = quote(query)
        response = SESSION.get(
            f"{USER_RETRIEVE_URL}?query={encoded_query}&k={k}",
            timeout=RETRIEVE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying retrieval service: {str(e)}")


def _user_path_matches(metadata: Dict[str, Any], user_id: str) -> bool:
    path = str(metadata.get("path", "")).replace("\\", "/").lower()
    token = f"/user_uploads/{user_id}/".lower()
    return token in path or path.startswith(f"user_uploads/{user_id}/".lower())

def _thread_path_matches(metadata: Dict[str, Any], thread_id: str) -> bool:
    path = str(metadata.get("path", "")).replace("\\", "/").lower()
    token = f"/uploads/{thread_id}/".lower()
    return token in path or path.startswith(f"uploads/{thread_id}/".lower())


def _filter_docs_for_user(documents: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        return documents
    filtered = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        if _user_path_matches(meta, user_id):
            filtered.append(doc)
    return filtered

def _filter_docs_for_thread(documents: List[Dict[str, Any]], thread_id: str) -> List[Dict[str, Any]]:
    if not thread_id:
        return documents
    filtered = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        if _thread_path_matches(meta, thread_id):
            filtered.append(doc)
    return filtered

def format_context(documents: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into a structured context string,
    including metadata like document name, page number, and file type.
    """
    formatted_docs = []
    
    for i, doc in enumerate(documents, 1):
        meta = doc.get("metadata", {})
        file_path = meta.get("path", "Unknown")
        file_name = os.path.basename(file_path)
        page_number = meta.get("page_number", "N/A")

        formatted_docs.append(
            f" **Name:** {file_name}\n"
            f" **Page:** {page_number}\n"
            f"---\n"
            f"{_truncate_text(doc.get('text', '').strip(), CONTEXT_TEXT_MAX_CHARS)}"
        )

    return "\n\n".join(formatted_docs)

def generate_answer_openai(query: str, source: str,retrieved_docs: List[Dict[str, Any]], max_tokens: int = 1000, model: str = "gpt-4o-mini") -> str:
    """
    Generate an answer using OpenAI model
    """
    try:
       
        context = format_context(retrieved_docs)
   
        client = get_client(model)
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise and factual 3GPP research  assistant. Answer questions based on the provided context.
                    Important Instructions:
                    1. Base your answer on the provided context documents
                    2. THINK and REASON out your analysis while generating response. Do comparative and critical analysis if required.  
                    3. Use quotes when directly quoting text
                    4. If you find conflicting information, point it out 
                    5. Mention the name of the documents and the page number from where the answer is extracted
                    6. DO NOT GIVE THE SCORES OF THE DOCUMENTS, REMOVE THE SCORES IF PRESENT"""

                },
                {
                    "role": "user",
                    "content": f"""Context:\n{context}\n\nQuestion: {query}\n\n ."""
                }
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=max_tokens,
            model=model
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer with OpenAI: {str(e)}")
        
@app.post("/generate", response_model=AnswerResponse)
def generate(query_request: Query):
    """
    Generate an answer for a given query using retrieved documents and OpenAI LLM.
    Returns only the final answer from the LLM.
    """
    start = time.time()

    user_id = query_request.user_id
    thread_id = query_request.thread_id
    num_docs = max(1, int(query_request.num_docs))
    # Choose the appropriate retrieval service based on the destination
    if query_request.destination == "user":
        retrieval_k = num_docs
        if user_id:
            retrieval_k = max(retrieval_k * USER_RETRIEVE_MULTIPLIER, USER_RETRIEVE_MIN_K)
            retrieval_k = min(retrieval_k, USER_RETRIEVE_MAX_K)
        retrieved_docs = query_retrieval_service2(query_request.query, retrieval_k)
    else:
        retrieved_docs = query_retrieval_service(query_request.query, num_docs)

    if user_id:
        retrieved_docs = _filter_docs_for_user(retrieved_docs, user_id)
        if not retrieved_docs:
            return AnswerResponse(
                answer="No relevant documents found for this user. Ask the user to upload relevant documents."
            )
    if thread_id:
        retrieved_docs = _filter_docs_for_thread(retrieved_docs, thread_id)
        if not retrieved_docs:
            return AnswerResponse(
                answer="No relevant documents found for this thread. Ask the user to continue the thread or run a broader search."
            )
    logger.debug("Retrieved %s docs before rerank", len(retrieved_docs))
    docs_for_rerank = retrieved_docs
    if RERANK_MAX_DOCS > 0:
        docs_for_rerank = _cap_docs(docs_for_rerank, max(RERANK_MAX_DOCS, num_docs))
    # Rerank the retrieved documents
    reranked_docs = rerank_documents(query_request.query, docs_for_rerank)
    context_limit = num_docs
    if CONTEXT_MAX_DOCS > 0:
        context_limit = min(context_limit, CONTEXT_MAX_DOCS)
    reranked_docs = _cap_docs(reranked_docs, context_limit)
    logger.debug("Using %s docs for context", len(reranked_docs))
    # Generate answer using OpenAI with reranked documents
    answer = generate_answer_openai(
        query_request.query, 
        query_request.source, 
        reranked_docs,
        query_request.max_tokens,
        query_request.model
    )
    
    end = time.time()
    logger.info("RAG request completed in %.2fs", end - start)
    return AnswerResponse(answer=answer)


if __name__ == "__main__":

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        'bind': '0.0.0.0:4005',
        'workers': 32,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 120,
        'graceful_timeout': 60,
        'keepalive': 5,      
    }
    StandaloneApplication(app, options).run()
