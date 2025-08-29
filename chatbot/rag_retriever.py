from glob import glob
import logging
import os
from uuid import uuid4
from flask import current_app as app
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from api.controllers.document_chunk_controller import DocumentChunkController
from chatbot.document_parsers import DocumentParsers

class RagRetriever:
    def __init__(self, chatbot_id=None, vector_db_path=None, documents_path=None):
        self.chatbot_id = chatbot_id
        # Suppress verbose logging
        logging.getLogger("unstructured").setLevel(logging.WARNING)
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings_function = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        self.chunk_size = 1200
        self.chunk_overlap = 120

        if vector_db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vector_db_path = os.path.join(project_root, "rag_milvus.db")

        URI = vector_db_path
        parse_documents = False

        if not os.path.exists(vector_db_path):
            parse_documents = True

        self.vector_store = Milvus(
            embedding_function=self.embeddings_function,
            connection_args={"uri": URI},
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            index_params=[
                {"index_type": "FLAT", "metric_type": "COSINE"},  # for dense vectors
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}  # for sparse vectors (BM25)
            ],
            consistency_level="Bounded",
            drop_old=False,
        )

        if parse_documents:
            if documents_path is None:
                documents_path = "./documents/"
            self.save_pdf_documents_at_path(documents_path)

    def process_documents(self, documents):
        """Process documents for text extraction"""
        text_documents = DocumentParsers.unstructured_parser(documents)
        return text_documents

    def chunk_documents(self, documents):
        """Chunk documents for storage in Milvus"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        # Ensure chunk metadata for tracking
        for chunk in chunks:
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = 'unknown'

        return chunks

    def generate_unique_uuids(self, document_count):
        """Generate unique UUIDs that don't already exist in the vector store"""
        uuids = []
        max_attempts = 3

        for _ in range(document_count):
            attempts = 0
            while attempts < max_attempts:
                new_uuid = str(uuid4())
                try:
                    # Check if the UUID already exists
                    existing = self.vector_store.get(ids=[new_uuid])
                    if not existing or len(existing) == 0:
                        # UUID doesn't exist, we can use it
                        uuids.append(new_uuid)
                        break
                except Exception:
                    uuids.append(new_uuid)
                    break
                attempts += 1
        return uuids

    def save_documents(self, documents):
        """Save and index a list of documents into the vector store.
            Params:
                documents (list): A list of document paths to be saved.
        """
        text_documents = self.process_documents(documents)
        chunks = self.chunk_documents(text_documents)

        # Save to vector store with unique UUIDs
        uuids = self.generate_unique_uuids(len(chunks))
        
        for chunk, uuid in zip(chunks, uuids):
            DocumentChunkController.create_document_chunk(self.chatbot_id, document_name=chunk.metadata['source'], uuid=uuid)

        # Add documents to vector store
        self.vector_store.add_documents(chunks, ids=uuids)
        print(f"Loaded and indexed {len(chunks)} document chunks")

    def delete_document(self, document_name):
        uuids = DocumentChunkController.get_document_chunks_uuids_by_document_name(self.chatbot_id, document_name)
        DocumentChunkController.delete_document_chunks_by_document_name(document_name, self.chatbot_id)
        self.vector_store.delete_documents(uuids)
        return uuids

    def save_pdf_documents_at_path(self, documents_path):
        pdf_documents = glob(f"{documents_path}/*.pdf")
        self.save_documents(pdf_documents)

    def invoke(self, query):
        # Retrieve documents based on the query
        # Rerank results using RRF
        results = self.vector_store.similarity_search(
            query, k=5, ranker_type="rrf"
        )

        return results