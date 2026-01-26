from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class AdaptiveChunker:
    """Implements multiple chunking strategies based on document type."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Sentence-level for general text
        self.sentence_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Semantic chunking for technical docs
        self.semantic_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=256,  # Fixed: all-MiniLM-L6-v2 max token limit is 256
            model_name="all-MiniLM-L6-v2"
        )
    
    def chunk_document(self, document: Document, doc_type: str = "general") -> List[Document]:
        """Select chunking strategy based on document characteristics."""
        if doc_type == "technical":
            return self.semantic_splitter.split_documents([document])
        elif doc_type == "legal":
            return self.proposition_chunking(document)
        elif doc_type == "code":
            return self.code_chunking(document)
        else:
            return self.sentence_splitter.split_documents([document])
    
    def proposition_chunking(self, document: Document) -> List[Document]:
        """Break down document into atomic propositions/facts."""
        text = document.page_content
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk += sentence + " "
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=document.metadata
                    ))
                current_chunk = sentence + " "
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=document.metadata
            ))
        
        return chunks
    
    def code_chunking(self, document: Document) -> List[Document]:
        """Chunk code documents preserving function/class boundaries."""
        text = document.page_content
        
        # Split by functions/classes
        function_pattern = r'(def\s+\w+|class\s+\w+|async\s+def\s+\w+)'
        sections = re.split(function_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for i, section in enumerate(sections):
            if i % 2 == 1:  # This is a function/class declaration
                section = section + sections[i + 1] if i + 1 < len(sections) else section
                
            if len(current_chunk + section) <= self.chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=document.metadata
                    ))
                current_chunk = section
        
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=document.metadata
            ))
        
        return chunks
