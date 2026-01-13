"""
Qdrant Vector Database Client for FluidOrbit

This module provides a client for interacting with the Qdrant vector database.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    UpdateStatus,
)
from typing import Optional, List, Dict, Any
import os


class QdrantDB:
    """
    Qdrant database client wrapper for FluidOrbit.
    
    Provides methods for:
    - Creating/managing collections
    - Upserting vectors with payloads
    - Searching similar vectors
    - Filtering and querying
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        url: str = None,
        api_key: str = None,
    ):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server host (default: localhost)
            port: Qdrant server port (default: 6333)
            url: Full Qdrant URL (overrides host/port if provided)
            api_key: API key for Qdrant Cloud (optional)
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(host=self.host, port=self.port)
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk: bool = False,
    ) -> bool:
        """
        Create a new collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (e.g., 1536 for OpenAI, 768 for BERT)
            distance: Distance metric (COSINE, EUCLID, DOT)
            on_disk: Whether to store vectors on disk (for large collections)
            
        Returns:
            True if collection was created, False if it already exists
        """
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]
        
        if collection_name in existing_names:
            return False
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
                on_disk=on_disk,
            ),
        )
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        return self.client.delete_collection(collection_name=collection_name)
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        collections = self.client.get_collections().collections
        return collection_name in [c.name for c in collections]
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.value,
        }
    
    def upsert_points(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]] = None,
        batch_size: int = 100,
    ) -> bool:
        """
        Upsert (insert or update) points into a collection.
        
        Args:
            collection_name: Target collection
            ids: List of point IDs (strings or ints)
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            batch_size: Number of points per batch
            
        Returns:
            True if successful
        """
        if payloads is None:
            payloads = [{}] * len(ids)
        
        points = [
            PointStruct(id=idx, vector=vec, payload=payload)
            for idx, vec, payload in zip(ids, vectors, payloads)
        ]
        
        # Batch upsert for better performance
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            result = self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            if result.status != UpdateStatus.COMPLETED:
                return False
        
        return True
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Dict[str, Any] = None,
        score_threshold: float = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Filter conditions (e.g., {"category": "electronics"})
            score_threshold: Minimum similarity score
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            
        Returns:
            List of search results with id, score, and payload
        """
        query_filter = None
        if filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload if with_payload else None,
                "vector": hit.vector if with_vectors else None,
            }
            for hit in results
        ]
    
    def delete_points(
        self,
        collection_name: str,
        ids: List[str] = None,
        filters: Dict[str, Any] = None,
    ) -> bool:
        """
        Delete points by IDs or filter.
        
        Args:
            collection_name: Target collection
            ids: List of point IDs to delete
            filters: Filter conditions for deletion
            
        Returns:
            True if successful
        """
        if ids:
            self.client.delete(
                collection_name=collection_name,
                points_selector=ids,
            )
        elif filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(must=conditions),
            )
        return True
    
    def count_points(self, collection_name: str) -> int:
        """Get the number of points in a collection."""
        info = self.client.get_collection(collection_name=collection_name)
        return info.points_count
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


# Singleton instance for easy import
_qdrant_instance: Optional[QdrantDB] = None


def get_qdrant_client() -> QdrantDB:
    """Get or create the Qdrant client singleton."""
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantDB()
    return _qdrant_instance


# Collection names constants
PRODUCTS_COLLECTION = "products"
CHUNKS_COLLECTION = "product_chunks"
