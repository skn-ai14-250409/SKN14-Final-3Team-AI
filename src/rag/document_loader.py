from typing import List
from pathlib import Path
import os
import io

import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile
import pdfplumber

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

CSV_EXTENSION = ".csv"
PDF_EXTENSION = ".pdf"

class DocumentLoader:
    async def get_document_chunks(self, file: UploadFile) -> List[Document]:
        content_bytes: bytes = await file.read()
        file_name: str = file.filename
        _, file_ext = os.path.splitext(file_name)

        if file_ext == CSV_EXTENSION:
            return self.get_csv_chunks(content_bytes)
        elif file_ext == PDF_EXTENSION:
            return self.get_pdf_chunks(content_bytes)
        raise Exception("Unable to chunk document.")

    def get_csv_chunks(self, content_bytes: bytes) -> List[Document]:
        loaded_csv = pd.read_csv(io.BytesIO(content_bytes))
        chunks: List[Document] = []
        for idx, row in tqdm(loaded_csv.iterrows()):
            chunks.append(Document(
                page_content=str(row)
            ))
        return chunks
    
    def get_pdf_chunks(self, content_bytes: bytes) -> List[Document]:
        text: str = ""
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        texts: List[str] = text_splitter.split_text(text)
        chunks: List[Document] = []
        for text in texts:
            chunks.append(Document(page_content=text))
        return chunks