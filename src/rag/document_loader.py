from typing import List
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

CSV_EXTENSION = ".csv"
PDF_EXTENSION = ".pdf"

class DocumentLoader:
    def get_document_chunks(self, path: str, category: str) -> List[Document]:
        file = Path(path)
        file_ext: str = file.suffix

        if file_ext == CSV_EXTENSION:
            return self.get_csv_chunks(path, category)
        elif file_ext == PDF_EXTENSION:
            return self.get_pdf_chunks(path, category)
        raise Exception("Unable to chunk document.")

    def get_csv_chunks(self, path: str, category: str) -> List[Document]:
        loaded_csv = pd.read_csv(path)
        chunks: List[Document] = []
        for idx, row in tqdm(loaded_csv.iterrows(), total=len(loaded_csv)):
            chunks.append(Document(
                page_content=str(row),
                metadata={
                    "category": category,
                    "file_name": Path(path).name,
                    "row": idx
                }
            ))
        return chunks
    
    def get_pdf_chunks(self, path: str, category: str) -> List[Document]:
        loader = PDFPlumberLoader(path)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks: List[Document] = loader.load_and_split(
            text_splitter
        )
        for i, chunk in enumerate(chunks):
            chunk.metadata["category"] = category
            chunk.metadata["file_name"] = Path(path).name
            chunk.metadata["chunk_index"] = i


        return chunks
        