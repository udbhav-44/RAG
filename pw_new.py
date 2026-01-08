import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import pathway as pw
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy
from pathway.xpacks.llm import embedders, llms, parsers, prompts
from pathway.xpacks.llm.question_answering import AdaptiveRAGQuestionAnswerer
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer
from pathway.stdlib.indexing import UsearchKnnFactory, TantivyBM25Factory, HybridIndexFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import threading
os.environ["PATHWAY_PERSISTENT_STORAGE"] = "./persistence_data"

load_dotenv('.env')

LOG_DIR = os.getenv("PW_NEW_LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "pw_new.log")


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
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


# Configure text splitting parameters for document chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
# /usr/share/tesseract-ocr/4.00/tessdata
# /opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata
class DocumentProcessor:
    """
    A class to process documents and manage a document store server.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        # Configure environment and logging
        os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata"
        logger.info("Initializing DocumentProcessor")
        
        self.host = host
        self.port = port
        self.app = None 
        self.vector_store = None
        
        self.parser = parsers.ParseUnstructured(mode="paged")
        self.embedder = embedders.LiteLLMEmbedder(
            capacity=100,
            model='voyage/voyage-3-large',
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
            cache_strategy=DiskCache(),
        )
        

    def initialize_vector_store(self, path1):
        """       
        Initialize document store with provided file path
        """
        logger.info("Initializing vector store from %s", path1)

        source1 = pw.io.fs.read(path=path1, with_metadata=True, format="binary", mode="streaming")  
        path2 = "../3GPP-pipeline/temp_rag_space"
        source2 = pw.io.fs.read(path=path2, with_metadata=True, format="binary", mode="streaming")


        usearch = UsearchKnnFactory(embedder=self.embedder)
        bm25 = TantivyBM25Factory(ram_budget=524288000, in_memory_index=True)
        factories = [usearch, bm25]
        retriever_factory = HybridIndexFactory(factories, k=60)
        
        self.vector_store = DocumentStore.from_langchain_components(
            retriever_factory=retriever_factory,
            docs=[source1,source2],

            parser=self.parser,
            splitter=text_splitter,
        )


    def setup_document_server(self):
        """
        Configure document store server
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        logger.info("Configuring document store server on %s:%s", self.host, 4004)
            
        self.app1 = DocumentStoreServer(
            host=self.host,
            port=4004,
            document_store=self.vector_store
        )


    def start_document_server(self):
        """
        Launch document server in background thread
        """
        logger.info("Starting document server thread")
        server_thread = threading.Thread(
            target=self.app1.run,
            name="BaseDocument"
        )   
        server_thread.daemon = True
        server_thread.start()
        logger.info("Document server thread started")
        return server_thread
    
def main():
    # Initialize data directory for document storage
    data_dir = "./uploads"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up and start servers
    processor = DocumentProcessor()
    processor.initialize_vector_store(data_dir)
    processor.setup_document_server()
    
    persistence_backend = pw.persistence.Backend.filesystem("./state/")
    persistence_config = pw.persistence.Config(persistence_backend)
    
    try:
        logger.info("Starting Pathway pipeline")
        pw.run()
        processor.start_document_server()
        logger.info("Pathway pipeline stopped")
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception:
        logger.exception("pw_new crashed")

if __name__ == "__main__":
    main()



