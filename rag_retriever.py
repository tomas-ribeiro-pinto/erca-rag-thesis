from glob import glob
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import pymupdf4llm


class RagRetriever:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="llama3.1")
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
            # Use pymupdf4llm to extract markdown content
            md_text = pymupdf4llm.to_markdown(pdf_file)
            
            # Create a LangChain Document object
            doc = Document(
                page_content=md_text,
                metadata={"source": pdf_file}
            )
            documents.append(doc)
        
        return documents

    async def save_documents(self, documents_path):
        # Save documents to a vector store
        documents = await self.process_documents(documents_path)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents, ids=uuids)
        print(f"Loaded {len(documents)} documents from {documents_path}")

    def invoke(self, query, k=5):
        # Retrieve documents based on the query
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        results = self.retriever.invoke(query)
        return results