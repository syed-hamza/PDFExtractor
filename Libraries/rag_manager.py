from typing import Optional, List, Dict
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from pathlib import Path

class RAGManager:
    def __init__(self, openai_api_key: str):
        """Initialize the RAG manager with OpenAI API key."""
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.openai_api_key = openai_api_key
        self.index_dir = "storage/indices"
        
        # Set OpenAI API key
        import openai
        openai.api_key = openai_api_key
        
        # Create LLM with API key
        self.llm = OpenAI(api_key=openai_api_key)
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
    
    def _get_index_path(self, doc_id: str) -> str:
        """Get the path where the index for a document should be stored."""
        return os.path.join(self.index_dir, doc_id)
    
    def is_indexed(self, doc_id: str) -> bool:
        """Check if a document has been indexed."""
        index_path = self._get_index_path(doc_id)
        return os.path.exists(index_path)
    
    def index_document(self, pdf_path: str, doc_id: str) -> bool:
        """Index a PDF document and store its index."""
        try:
            # Create service context with OpenAI
            service_context = ServiceContext.from_defaults(llm=self.llm)
            
            # Load document
            documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context
            )
            
            # Save index
            index.storage_context.persist(persist_dir=self._get_index_path(doc_id))
            return True
            
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False
    
    def query_document(self, doc_id: str, query: str) -> Optional[str]:
        """Query an indexed document."""
        try:
            if not self.is_indexed(doc_id):
                return "Document is not indexed yet. Please index it first."
            
            # Load the existing index
            service_context = ServiceContext.from_defaults(llm=self.llm)
            storage_context = StorageContext.from_defaults(
                persist_dir=self._get_index_path(doc_id)
            )
            index = load_index_from_storage(
                storage_context,
                service_context=service_context
            )
            
            # Query the index
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            
            return str(response)
            
        except Exception as e:
            print(f"Error querying document: {e}")
            return None
    
    def remove_index(self, doc_id: str) -> bool:
        """Remove the index for a document."""
        try:
            index_path = self._get_index_path(doc_id)
            if os.path.exists(index_path):
                import shutil
                shutil.rmtree(index_path)
            return True
        except Exception as e:
            print(f"Error removing index: {e}")
            return False 