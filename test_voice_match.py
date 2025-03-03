"""Tests for voice matching functionality."""
import pytest
import numpy as np
from voice_match import VoiceMatcher, MatchResult

def generate_random_embedding(dim=192):
    """Generate a random normalized embedding vector."""
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)

def test_voice_matcher_init():
    """Test VoiceMatcher initialization."""
    # Test valid initialization
    ref = generate_random_embedding()
    matcher = VoiceMatcher(ref)
    assert matcher.dim == 192
    assert np.allclose(np.linalg.norm(matcher.ref), 1.0)
    
    # Test list input
    ref_list = ref.tolist()
    matcher = VoiceMatcher(ref_list)
    assert isinstance(matcher.ref, np.ndarray)
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        VoiceMatcher(np.array([]))  # Empty
    with pytest.raises(ValueError):
        VoiceMatcher(np.array([np.nan, np.inf]))  # Invalid values

def test_compute_similarity():
    """Test similarity computation."""
    ref = generate_random_embedding()
    matcher = VoiceMatcher(ref)
    
    # Test self-similarity
    assert np.isclose(matcher.compute_similarity(ref), 1.0)
    
    # Test orthogonal vectors
    ortho = np.zeros_like(ref)
    ortho[0] = 1.0
    ref[0] = 0.0
    matcher = VoiceMatcher(ref)
    assert np.isclose(matcher.compute_similarity(ortho), 0.0)
    
    # Test dimension mismatch
    with pytest.raises(ValueError):
        matcher.compute_similarity(np.random.rand(100))

def test_is_match():
    """Test match checking."""
    ref = generate_random_embedding()
    matcher = VoiceMatcher(ref)
    
    # Test exact match
    result = matcher.is_match(ref)
    assert isinstance(result, MatchResult)
    assert result.is_match is True
    assert np.isclose(result.similarity, 1.0)
    
    # Test clear non-match
    non_match = -ref  # Opposite direction
    result = matcher.is_match(non_match)
    assert result.is_match is False
    assert np.isclose(result.similarity, -1.0)
    
    # Test threshold behavior
    result = matcher.is_match(ref, threshold=0.99)
    assert result.is_match is True
    
    # Test invalid threshold
    with pytest.raises(ValueError):
        matcher.is_match(ref, threshold=1.5)
    with pytest.raises(ValueError):
        matcher.is_match(ref, threshold=-0.5)

def test_find_best_match():
    """Test finding best match from list."""
    ref = generate_random_embedding()
    matcher = VoiceMatcher(ref)
    
    # Generate test embeddings
    embeddings = [
        ref,  # Exact match
        -ref,  # Opposite
        generate_random_embedding(),  # Random
        ref * 0.9  # Similar
    ]
    
    result = matcher.find_best_match(embeddings)
    assert result is not None
    assert result.is_match is True
    assert np.isclose(result.similarity, 1.0)
    
    # Test empty list
    assert matcher.find_best_match([]) is None

def test_false_positive_rate():
    """Test false positive rate with random embeddings."""
    ref = generate_random_embedding()
    matcher = VoiceMatcher(ref)
    
    # Generate 1000 random embeddings
    n_tests = 1000
    false_positives = 0
    threshold = 0.8
    
    for _ in range(n_tests):
        test_emb = generate_random_embedding()
        result = matcher.is_match(test_emb, threshold=threshold)
        if result.is_match:
            false_positives += 1
    
    false_positive_rate = false_positives / n_tests
    assert false_positive_rate < 0.05, f"False positive rate too high: {false_positive_rate:.3f}"
    print(f"\nFalse positive rate: {false_positive_rate:.3f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=voice_match", "--cov-report=term-missing"]) 