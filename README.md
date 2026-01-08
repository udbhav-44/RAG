# Introduction  

This entire folder contains a full fledged RAG pipeline implemented using pathway's engine and its classes. It includes parsing , chunking , embedding , retrieving , reranking and response generation along with support for the same on user uploaded documents .
![Pathway](../assets/pway_rag.png)

## Pathway Instances to support tool calls and user uploads   

``pw_new.py`` starts a pathway DocumentStoreServer instance that specifically indexes and monitors a folder that saves .txt files saved by web scrapers , legal research tools like Indian Kanoon, financial tools like get_sec_filings.  It indexes them into the document store and allows for realtime questioning and answering for these documents as they are fetched by the tool calls in our agentic pipeline. 
Whereas ``pw_userkb.py`` starts the same but monitors the user uploads folder.

Both of them have 3 endpoints and run on their separate ports (4004 and 4006)  : 

1. `/v1/retrieve`: Retrieves documents based on a query (has query and k as parameters)
2. `/v1/statistics`: Provides statistics about the document store
3. `/v1/inputs`: Lists all documents in the store 

## Running and restart behavior

`start.py` is a simple runner that restarts `pw_new.py` if it exits with a non-zero code. It uses an exponential backoff between restarts. By default it runs the virtualenv at `./test/bin/python3`, so adjust `VENV_PYTHON` if your env lives elsewhere.

Logging:
- `pw_new.py` logs to `./logs/pw_new.log` (and stdout).
- `start.py` logs to `./logs/pw_new_runner.log`.

Tunable env vars:
- `PW_NEW_LOG_DIR` to change the log folder.
- `PW_NEW_RESTART_DELAY` and `PW_NEW_RESTART_MAX_DELAY` to control restart backoff (seconds).


## User Upload Server
``http_serve.py`` starts a FastAPI upload service on port ``8000`` with per-user isolation. Files are stored under ``user_uploads/<user_id>/`` based on the JWT ``sub`` (or email) claim. Endpoints:

- ``POST /upload``: Upload a file (multipart form-data, field ``file``).
- ``GET /uploads``: List files for the authenticated user.
- ``DELETE /upload/{filename}``: Delete a file owned by the authenticated user.

Auth: include ``Authorization: Bearer <token>``. If ``AUTH_REQUIRE_TOKEN=true`` is set, unauthenticated requests are rejected.

## RAG server  :
``rag_server.py`` starts a server that has single endpoint ``/generate`` which based on the query retrives relevant chunks from any of the 2 document store servers based on a routing logic. When ``destination="user"`` and ``user_id`` is provided, results are filtered to the user's upload directory only.

Sample curl request to make a simple get request to generate a response.

```
curl -X POST "http://localhost:4005/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "what are the net product sales of AMZN in 2022 ?"}'

```
