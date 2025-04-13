import numpy as np
from .bert_model import get_tokenizer_model

# Get model from bert_model.py
_, model = get_tokenizer_model()

def get_embedding(text):
    """Get embedding for a single text string."""
    if not text or text.isspace():
        # Return zero vector with correct dimensionality
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Encode the text directly with the model
    try:
        embedding = model.encode(text, show_progress_bar=False)
        return embedding
    except Exception as e:
        print(f"Error encoding text: {e}")
        return np.zeros(model.get_sentence_embedding_dimension())

def get_long_text_embedding(text, chunk_size=512):
    """Get embedding for long text by chunking and averaging."""
    if not text or text.isspace():
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # For empty chunks
    if not chunks:
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Encode all chunks at once (more efficient)
    try:
        embeddings = model.encode(chunks, show_progress_bar=False)
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"Error with batch encoding: {e}")
        # Fallback to individual encoding
        try:
            embeddings = [get_embedding(chunk) for chunk in chunks]
            return np.mean(embeddings, axis=0)
        except Exception as e:
            print(f"Error with individual encoding: {e}")
            return np.zeros(model.get_sentence_embedding_dimension())