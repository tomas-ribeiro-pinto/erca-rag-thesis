from concurrent.futures import ThreadPoolExecutor
from glob import glob
import logging
import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbot.retrievers import Retrievers

class RagRetriever:
    def __init__(self, db_path=None):
        # Suppress verbose logging
        logging.getLogger("unstructured").setLevel(logging.WARNING)
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        self.chunk_size = 1024
        self.chunk_overlap = 15

        # Use absolute path to avoid working directory issues
        if db_path is None:
            # Default to project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(project_root, "rag_milvus.db")
        
        URI = db_path
        parse_documents = False

        if not os.path.exists(db_path):
            parse_documents = True

        self.vector_store = Milvus(
            embedding_function= self.embeddings,
            connection_args={"uri": URI},
            index_params={"index_type": "FLAT", "metric_type": "L2"},
        )

        if parse_documents:
            documents_path = "./documents/"
            self.save_documents(documents_path)

    def process_documents(self, documents_path):
        pdf_files = glob(f"{documents_path}/*.pdf")
        #documents = Retrievers.pypdf_retriever(pdf_files)
        documents = Retrievers.unstructured_retriever(pdf_files)
        return documents
        
    def chunk_documents(self, documents):
        """Chunk documents for better embedding and storage in Milvus"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def save_documents(self, documents_path):
        # Save documents to a vector store
        documents = self.process_documents(documents_path)
        documents = self.chunk_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents, ids=uuids)
        print(f"Loaded {len(documents)} documents from {documents_path}")

    def invoke(self, query, k=5):
        # Retrieve documents based on the query
        self.retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
        results = self.retriever.invoke(query)
        return results