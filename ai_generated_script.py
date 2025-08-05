import os
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

class LocalPDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self) -> list[Document]:
        """Load PDF file and return a list of Document objects."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"The file {self.pdf_path} does not exist.")
        
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        return documents