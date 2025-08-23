from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

class DocumentParsers:
    @staticmethod
    def pypdf_parser(pdf_files):
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            for doc in docs:
                # Clean metadata
                cleaned_metadata = {
                    "source": doc.metadata.get("source", pdf_file),
                    "file_type": "pdf",
                    "page_number": doc.metadata.get("page_number", 0)
                }
                doc.metadata = cleaned_metadata
            documents.extend(docs)
        return documents
    
    @staticmethod
    def unstructured_parser(pdf_files):
        documents = []
        for pdf_file in pdf_files:
            loader_local = UnstructuredLoader(
                file_path=pdf_file,
                strategy="hi_res",
            )
            doc_local = ""
            for doc in loader_local.lazy_load():
                doc_local += doc.page_content
                print(f"Loaded document: {doc.metadata['source']}, Page: {doc.metadata.get('page_number', 0)}")

            combined_doc = Document(
                page_content=doc_local,
                metadata={
                    "source": pdf_file,
                    "file_type": "pdf",
                }
            )
            documents.append(combined_doc)

        return documents
