"""Voice similarity matching module."""
import numpy as np
import logging
from typing import Union, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Result of a voice similarity check."""
    is_match: bool
    similarity: float
    threshold: float

class VoiceMatcher:
    """Matches voice embeddings using cosine similarity."""
    
    def __init__(self, reference_embedding: np.ndarray):
        """Initialize with a reference voice embedding.
        
        Args:
            reference_embedding: Reference voice embedding vector
            
        Raises:
            ValueError: If reference embedding is invalid
        """
        if not isinstance(reference_embedding, np.ndarray):
            reference_embedding = np.array(reference_embedding)
            
        if not reference_embedding.size:
            raise ValueError("Reference embedding cannot be empty")
            
        if not np.isfinite(reference_embedding).all():
            raise ValueError("Reference embedding contains invalid values")
            
        # Normalize reference embedding
        self.ref = reference_embedding / np.linalg.norm(reference_embedding)
        self.dim = reference_embedding.size
        logger.debug(f"Initialized matcher with {self.dim}-dimensional reference")
    
    def compute_similarity(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity with reference embedding.
        
        Args:
            embedding: Voice embedding to compare
            
        Returns:
            float: Cosine similarity score in range [-1, 1]
            
        Raises:
            ValueError: If embedding dimensions don't match
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        if embedding.size != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {embedding.size}, "
                f"expected {self.dim}"
            )
            
        # Normalize query embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Compute cosine similarity
        similarity = np.dot(self.ref, embedding)
        
        return float(similarity)  # Convert to Python float
    
    def is_match(self, embedding: np.ndarray, threshold: float = 0.8) -> Union[bool, MatchResult]:
        """Check if embedding matches the reference voice.
        
        Args:
            embedding: Voice embedding to compare
            threshold: Similarity threshold (0.8 default)
            
        Returns:
            bool: True if similarity >= threshold
            
        Raises:
            ValueError: If embedding is invalid or threshold out of range
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
            
        try:
            similarity = self.compute_similarity(embedding)
            is_match = similarity >= threshold
            
            logger.debug(
                f"Match result: {is_match} "
                f"(similarity: {similarity:.3f}, threshold: {threshold})"
            )
            
            return MatchResult(
                is_match=is_match,
                similarity=similarity,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error matching voice: {str(e)}")
            raise
    
    def find_best_match(self, embeddings: list[np.ndarray]) -> Optional[MatchResult]:
        """Find the best matching embedding from a list.
        
        Args:
            embeddings: List of voice embeddings to compare
            
        Returns:
            MatchResult: Best match result or None if no embeddings
        """
        if not embeddings:
            return None
            
        similarities = [self.compute_similarity(emb) for emb in embeddings]
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        return MatchResult(
            is_match=best_sim >= 0.8,  # Use default threshold
            similarity=best_sim,
            threshold=0.8
        ) 