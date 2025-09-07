import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct, 
    Filter, FieldCondition, SearchRequest
)


class QdrantVectorStore:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        url: str = None,
        api_key: str = None,
        collection_name: str = "rag_documents"
    ):
        self.collection_name = collection_name
        
        # Use environment variables as defaults
        if host is None:
            host = os.environ.get("QDRANT_HOST", "localhost")
        if port is None:
            port = int(os.environ.get("QDRANT_PORT", "6333"))
        if url is None:
            url = os.environ.get("QDRANT_URL")
        if api_key is None:
            api_key = os.environ.get("QDRANT_API_KEY")
            
        # Initialize client
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)
    
    def create_collection(self, vector_size: int, distance: Distance = Distance.COSINE):
        """Create a new collection with specified vector size."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            print(f"✅ Created collection '{self.collection_name}' with vector size {vector_size}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Collection '{self.collection_name}' already exists")
            else:
                raise e
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✅ Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def upsert_vectors(
        self, 
        vectors: np.ndarray, 
        payloads: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ):
        """Insert or update vectors with metadata payloads."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        points = []
        for i, (vector, payload) in enumerate(zip(vectors, payloads)):
            # Generate UUID for point ID and store original ID in payload if provided
            point_id = str(uuid.uuid4())
            if ids and i < len(ids):
                payload["original_id"] = ids[i]
            
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload
            ))
        
        # Batch upsert in chunks to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"✅ Upserted {len(points)} vectors to collection '{self.collection_name}'")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Convert numpy array to list
        if isinstance(query_vector, np.ndarray):
            if query_vector.ndim > 1:
                query_vector = query_vector.flatten()
            query_vector = query_vector.tolist()
        
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match={"value": value}))
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        )
        
        # Format results
        hits = []
        for result in results:
            hit = {
                "id": result.id,
                "score": float(result.score),
                **result.payload
            }
            hits.append(hit)
        
        return hits
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                }
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def scroll_all_points(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all points from the collection."""
        all_points = []
        offset = None
        
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not results:
                break
                
            for point in results:
                all_points.append({
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                })
            
            offset = next_offset
            if offset is None:
                break
        
        return all_points
    
    def count_points(self) -> int:
        """Count total points in collection."""
        info = self.get_collection_info()
        return info.get("points_count", 0)


def get_qdrant_store(collection_name: str = "rag_documents") -> QdrantVectorStore:
    """Factory function to create QdrantVectorStore with environment configuration."""
    return QdrantVectorStore(collection_name=collection_name)