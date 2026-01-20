import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import threading

import pathway as pw
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy
from pathway.xpacks.llm import embedders, parsers
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer
from pathway.stdlib.indexing import UsearchKnnFactory, TantivyBM25Factory, HybridIndexFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(".env")

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_mode(value: str, default: str = "streaming") -> str:
    value = (value or "").strip().lower()
    if value in {"streaming", "static"}:
        return value
    return default


PERSIST_DIR = os.getenv("PW_NEW_PERSIST_DIR", "./persistence_data")
os.environ.setdefault("PATHWAY_PERSISTENT_STORAGE", PERSIST_DIR)
CACHE_DIR = os.getenv("PW_NEW_CACHE_DIR", PERSIST_DIR)

LOG_DIR = os.getenv("PW_NEW_LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "pw_new.log")
LOG_LEVEL = os.getenv("PW_NEW_LOG_LEVEL", "INFO").upper()

HOST = os.getenv("PW_NEW_HOST", "0.0.0.0")
PORT = _env_int("PW_NEW_PORT", 4004)

DATA_DIR = os.getenv("PW_NEW_DATA_DIR", "./uploads")
TEMP_DIR = os.getenv("PW_NEW_TEMP_DIR", "../3GPP-pipeline/temp_rag_space")
UPLOAD_MODE = _normalize_mode(os.getenv("PW_NEW_UPLOAD_MODE", "streaming"))
TEMP_MODE = _normalize_mode(os.getenv("PW_NEW_TEMP_MODE", "streaming"))
OBJECT_PATTERN = os.getenv("PW_NEW_OBJECT_PATTERN", "*")
AUTOCOMMIT_MS = _env_int("PW_NEW_AUTOCOMMIT_MS", 500)

CHUNK_SIZE = _env_int("PW_NEW_CHUNK_SIZE", 4000)
CHUNK_OVERLAP = _env_int("PW_NEW_CHUNK_OVERLAP", 800)
DISABLE_SPLIT = _env_bool("PW_NEW_DISABLE_SPLIT", False)

PARSER_MODE = os.getenv("PW_NEW_PARSER_MODE", "paged")
PARSER_STRATEGY = os.getenv("PW_NEW_PARSER_STRATEGY", "").strip()

EMBED_MODEL = os.getenv("PW_NEW_EMBED_MODEL", "voyage/voyage-3-large")
EMBED_CAPACITY = _env_int("PW_NEW_EMBED_CAPACITY", 100)
EMBED_CACHE_SIZE = _env_int("PW_NEW_EMBED_CACHE_SIZE", 2**30)
PARSER_CACHE_SIZE = _env_int("PW_NEW_PARSER_CACHE_SIZE", 2**30)

INDEX_MODE = os.getenv("PW_NEW_INDEX_MODE", "hybrid").strip().lower()
RRF_K = _env_float("PW_NEW_RRF_K", 60.0)
BM25_RAM_BUDGET = _env_int("PW_NEW_BM25_RAM_BUDGET", 524_288_000)
BM25_IN_MEMORY = _env_bool("PW_NEW_BM25_IN_MEMORY", True)
ENABLE_CACHE = _env_bool("PW_NEW_ENABLE_CACHE", True)

USEARCH_RESERVED = _env_int("PW_NEW_USEARCH_RESERVED", 400)
USEARCH_CONNECTIVITY = _env_int("PW_NEW_USEARCH_CONNECTIVITY", 0)
USEARCH_EXPANSION_ADD = _env_int("PW_NEW_USEARCH_EXPANSION_ADD", 0)
USEARCH_EXPANSION_SEARCH = _env_int("PW_NEW_USEARCH_EXPANSION_SEARCH", 0)

MAX_DOC_CHARS = _env_int("PW_NEW_MAX_DOC_CHARS", 0)


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def _log_uncaught_exceptions(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logging.getLogger("pw_new").exception("Uncaught exception", exc_info=(exc_type, exc, tb))


def _log_thread_exceptions(args):
    logging.getLogger("pw_new").exception(
        "Thread crashed: %s", args.thread.name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


setup_logging()
sys.excepthook = _log_uncaught_exceptions
if hasattr(threading, "excepthook"):
    threading.excepthook = _log_thread_exceptions

logger = logging.getLogger("pw_new")


if DISABLE_SPLIT or CHUNK_SIZE <= 0:
    text_splitter = None
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

class DocumentProcessor:
    """
    A class to process documents and manage a document store server.
    """

    def __init__(self, host: str = HOST, port: int = PORT):
        tessdata_prefix = os.getenv("TESSDATA_PREFIX")
        if not tessdata_prefix:
            for candidate in (
                "/usr/share/tesseract-ocr/4.00/tessdata",
                "/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata",
            ):
                if os.path.isdir(candidate):
                    os.environ["TESSDATA_PREFIX"] = candidate
                    break
        logger.info("Initializing DocumentProcessor (index_mode=%s)", INDEX_MODE)
        
        self.host = host
        self.port = port
        self.app = None 
        self.vector_store = None
        self.index_mode = INDEX_MODE if INDEX_MODE in {"hybrid", "bm25", "vector"} else "hybrid"
        if self.index_mode != INDEX_MODE:
            logger.warning("Unknown PW_NEW_INDEX_MODE=%s; defaulting to hybrid", INDEX_MODE)

        parser_mode = PARSER_MODE if PARSER_MODE in {"single", "elements", "paged"} else "paged"
        if parser_mode != PARSER_MODE:
            logger.warning("Unknown PW_NEW_PARSER_MODE=%s; defaulting to paged", PARSER_MODE)

        parser_kwargs = {}
        if PARSER_STRATEGY:
            parser_kwargs["strategy"] = PARSER_STRATEGY

        self.parser = parsers.ParseUnstructured(
            mode=parser_mode,
            cache_strategy=DiskCache(size_limit=PARSER_CACHE_SIZE),
            **parser_kwargs,
        )

        self.embedder = None
        if self.index_mode in {"hybrid", "vector"}:
            self.embedder = embedders.LiteLLMEmbedder(
                capacity=EMBED_CAPACITY,
                model=EMBED_MODEL,
                retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
                cache_strategy=DiskCache(size_limit=EMBED_CACHE_SIZE),
            )

    def initialize_vector_store(self, path1):
        """       
        Initialize document store with provided file path
        """
        data_dir = os.path.abspath(path1)
        os.makedirs(data_dir, exist_ok=True)
        logger.info("Initializing vector store from %s", data_dir)

        source1 = pw.io.fs.read(
            path=data_dir,
            with_metadata=True,
            format="binary",
            mode=UPLOAD_MODE,
            object_pattern=OBJECT_PATTERN,
            autocommit_duration_ms=AUTOCOMMIT_MS,
            persistent_id="pw_new_uploads",
        )

        sources = [source1]
        if TEMP_DIR:
            temp_dir = os.path.abspath(TEMP_DIR)
            os.makedirs(temp_dir, exist_ok=True)
            source2 = pw.io.fs.read(
                path=temp_dir,
                with_metadata=True,
                format="binary",
                mode=TEMP_MODE,
                object_pattern=OBJECT_PATTERN,
                autocommit_duration_ms=AUTOCOMMIT_MS,
                persistent_id="pw_new_temp",
            )
            sources.append(source2)
        logger.info(
            "Sources=%s upload_mode=%s temp_mode=%s autocommit_ms=%s chunk_size=%s overlap=%s",
            len(sources),
            UPLOAD_MODE,
            TEMP_MODE,
            AUTOCOMMIT_MS,
            CHUNK_SIZE if text_splitter else 0,
            CHUNK_OVERLAP if text_splitter else 0,
        )

        if self.index_mode == "bm25":
            retriever_factory = TantivyBM25Factory(
                ram_budget=BM25_RAM_BUDGET, in_memory_index=BM25_IN_MEMORY
            )
        else:
            usearch = UsearchKnnFactory(
                embedder=self.embedder,
                reserved_space=USEARCH_RESERVED,
                connectivity=USEARCH_CONNECTIVITY,
                expansion_add=USEARCH_EXPANSION_ADD,
                expansion_search=USEARCH_EXPANSION_SEARCH,
            )
            if self.index_mode == "vector":
                retriever_factory = usearch
            else:
                bm25 = TantivyBM25Factory(
                    ram_budget=BM25_RAM_BUDGET, in_memory_index=BM25_IN_MEMORY
                )
                retriever_factory = HybridIndexFactory([usearch, bm25], k=RRF_K)

        doc_post_processors = []
        if MAX_DOC_CHARS > 0:
            def _truncate_doc(text: str, metadata: dict):
                if len(text) > MAX_DOC_CHARS:
                    return text[:MAX_DOC_CHARS], {**metadata, "truncated": True}
                return text, metadata

            doc_post_processors.append(_truncate_doc)
        
        self.vector_store = DocumentStore.from_langchain_components(
            retriever_factory=retriever_factory,
            docs=sources,
            parser=self.parser,
            splitter=text_splitter,
            doc_post_processors=doc_post_processors or None,
        )


    def setup_document_server(self):
        """
        Configure document store server
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        logger.info("Configuring document store server on %s:%s", self.host, self.port)
            
        self.app1 = DocumentStoreServer(
            host=self.host,
            port=self.port,
            document_store=self.vector_store
        )


    def run(self):
        if not self.app1:
            raise ValueError("Document server not configured")
        cache_backend = pw.persistence.Backend.filesystem(CACHE_DIR)
        logger.info(
            "Starting DocumentStoreServer on %s:%s (cache=%s, index_mode=%s)",
            self.host,
            self.port,
            ENABLE_CACHE,
            self.index_mode,
        )
        self.app1.run(with_cache=ENABLE_CACHE, cache_backend=cache_backend)
    
def main():
    # Initialize data directory for document storage
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    if TEMP_DIR:
        os.makedirs(os.path.abspath(TEMP_DIR), exist_ok=True)
    
    # Set up and start servers
    processor = DocumentProcessor()
    processor.initialize_vector_store(data_dir)
    processor.setup_document_server()
    
    try:
        logger.info("Starting Pathway pipeline")
        processor.run()
        logger.info("Pathway pipeline stopped")
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception:
        logger.exception("pw_new crashed")

if __name__ == "__main__":
    main()
