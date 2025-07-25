from concurrent.futures import ThreadPoolExecutor
from glob import glob
import logging
import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pymupdf4llm
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RagRetriever:
    def __init__(self):
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

        URI = "./rag_milvus.db"

        self.vector_store = Milvus(
            embedding_function= self.embeddings,
            connection_args={"uri": URI},
            index_params={"index_type": "FLAT", "metric_type": "L2"},
        )

    async def process_documents(self, documents_path):
        pdf_files = glob(f"{documents_path}/*.pdf")

        documents = []
        for pdf_file in pdf_files:
            # loader_local = UnstructuredLoader(
            #     file_path=pdf_file,
            #     strategy="hi_res",
            # )
            # docs_local = []
            # for doc in loader_local.lazy_load():
            #     # Clean metadata to avoid Milvus datatype issues
            #     cleaned_metadata = {
            #         "source": doc.metadata.get("source", pdf_file),
            #         "file_type": "pdf",
            #         "page_number": doc.metadata.get("page_number", 0)
            #     }
            #     doc.metadata = cleaned_metadata
            #     docs_local.append(doc)
            # documents.extend(docs_local)

            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            for doc in docs:
                # Clean metadata to avoid Milvus datatype issues
                cleaned_metadata = {
                    "source": doc.metadata.get("source", pdf_file),
                    "file_type": "pdf",
                    "page_number": doc.metadata.get("page_number", 0)
                }
                doc.metadata = cleaned_metadata
            documents.extend(docs)
            
        return documents
        
    def chunk_documents(self, documents):
        """Chunk documents for better embedding and storage in Milvus"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    async def save_documents(self, documents_path):
        # Save documents to a vector store
        documents = await self.process_documents(documents_path)
        documents = self.chunk_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents, ids=uuids)
        print(f"Loaded {len(documents)} documents from {documents_path}")

    def invoke(self, query, k=5):
        # Retrieve documents based on the query
        self.retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
        results = self.retriever.invoke(query)
        return results