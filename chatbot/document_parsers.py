from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader

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
            docs_local = []
            for doc in loader_local.lazy_load():
                # Clean metadata to avoid Milvus datatype issues
                cleaned_metadata = {
                    "source": doc.metadata.get("source", pdf_file),
                    "file_type": "pdf",
                    "page_number": doc.metadata.get("page_number", 0)
                }
                doc.metadata = cleaned_metadata
                docs_local.append(doc)
                print(f"Loaded document: {doc.metadata['source']}, Page: {doc.metadata['page_number']}")
            documents.extend(docs_local)

            # Save documents to a txt file
            with open("retrieved_documents.txt", "a", encoding="utf-8") as f:
                for doc in docs_local:
                    f.write(f"Source: {doc.metadata['source']}\n")
                    f.write(f"File Type: {doc.metadata['file_type']}\n")
                    f.write(f"Page Number: {doc.metadata['page_number']}\n")
                    f.write(f"Content:\n{doc.page_content}\n")
                    f.write("="*40 + "\n")

        return documents
