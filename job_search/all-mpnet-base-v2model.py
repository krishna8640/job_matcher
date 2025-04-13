from sentence_transformers import SentenceTransformer

# Choose one of these models:
# - 'all-mpnet-base-v2': Best quality (86.4% on STS benchmark), 768 dimensions
# - 'all-MiniLM-L6-v2': Better efficiency (80.9% on STS benchmark), 384 dimensions
MODEL_NAME = 'all-mpnet-base-v2'

# Load the sentence transformer model
try:
    sentence_model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded Sentence Transformer model: {MODEL_NAME}")
    print(f"Embedding dimension: {sentence_model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"Error loading primary model: {e}")
    try:
        # Fallback to smaller model if main one fails
        MODEL_NAME = 'all-MiniLM-L6-v2'
        sentence_model = SentenceTransformer(MODEL_NAME)
        print(f"Loaded fallback model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading fallback model: {e}")
        raise RuntimeError("Failed to load Sentence Transformer models")

def get_tokenizer_model():
    # For compatibility with original API
    return None, sentence_model