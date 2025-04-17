"""
FAISS index cache singleton to avoid rebuilding the index on every query.
"""

import faiss
import numpy as np
import tempfile
import os
import threading
import traceback
from .db import get_db_connection
from .config import FAISS_INDEX_NAME

_lock = threading.Lock()

def deserialize_faiss_index(serialized_index: bytes) -> faiss.Index:
    """Deserialize a FAISS index from bytes."""
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(serialized_index)
        return faiss.read_index(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

class IndexCache:
    """Singleton class to cache the FAISS index in memory."""
    
    _instance = None
    index: faiss.Index = None
    id_mapping: dict[int, str] = {}
    is_loaded: bool = False
    
    @classmethod
    def get_instance(cls) -> "IndexCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_index(self) -> None:
        """Load the pre-built FAISS index from the database (thread‑safe)."""
        with _lock:
            if self.is_loaded:
                return
            
            print(f"Loading FAISS index named `{FAISS_INDEX_NAME}` from database...")
            conn, cursor = get_db_connection()
            try:
                # 1) Fetch the serialized index blob
                cursor.execute(
                    "SELECT index_data FROM faiss_indices WHERE name = %s;",
                    (FAISS_INDEX_NAME,)
                )
                row = cursor.fetchone()
                if not row:
                    print(f"No FAISS index named `{FAISS_INDEX_NAME}` found — building fallback.")
                    return self._build_fallback_index()
                
                blob = row[0]
                print(f" - Retrieved {len(blob)} bytes of index data.")
                
                # 2) Deserialize
                self.index = deserialize_faiss_index(blob)
                print(" - Deserialized FAISS index successfully.")
                
                # 3) Load ID mapping
                cursor.execute(
                    "SELECT vector_position, job_id FROM faiss_job_mapping WHERE faiss_index_name = %s;",
                    (FAISS_INDEX_NAME,)
                )
                mapping_rows = cursor.fetchall()
                print(f" - Fetched {len(mapping_rows)} mapping rows.")
                
                self.id_mapping = {}
                for pos, jid in mapping_rows:
                    try:
                        pos_int = int(pos)
                        # Normalize job_id to string so downstream code doesn’t need to cast
                        self.id_mapping[pos_int] = str(jid)
                    except Exception as e:
                        print(f"   ⚠️ Warning: skipping mapping ({pos!r}, {jid!r}): {e}")
                
                self.is_loaded = True
                print(f"→ FAISS index cache ready with {len(self.id_mapping)} vectors.")
            
            except Exception:
                print("❌ Error loading FAISS index from DB, building fallback…")
                traceback.print_exc()
                self._build_fallback_index()
            
            finally:
                cursor.close()
                conn.close()
    
    def _build_fallback_index(self) -> None:
        """Fallback: build an IndexFlatL2 from existing embeddings."""
        print("Building fallback FAISS index from job_postings.embeddings …")
        conn, cursor = get_db_connection()
        try:
            cursor.execute("SELECT job_id, embedding FROM job_postings WHERE embedding IS NOT NULL;")
            rows = cursor.fetchall()
            if not rows:
                print(" - No embeddings in DB, cannot build fallback.")
                return
            
            embeddings = []
            ids = []
            for jid, emb in rows:
                try:
                    if isinstance(emb, (bytes, bytearray)):
                        vec = np.frombuffer(emb, dtype="float32")
                    elif isinstance(emb, str):
                        vals = [float(x) for x in emb.strip("[]").split(",")]
                        vec = np.array(vals, dtype="float32")
                    else:
                        vec = np.array(emb, dtype="float32")
                    
                    ids.append(str(jid))
                    embeddings.append(vec)
                
                except Exception as e:
                    print(f"   ⚠️ Skipping embedding for job_id={jid}: {e}")
            
            if not embeddings:
                print(" - After filtering, no valid embeddings remain.")
                return
            
            arr = np.vstack(embeddings).astype("float32")
            d = arr.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(arr)
            
            # build a simple 0→N‐1 positional mapping
            self.id_mapping = {i: ids[i] for i in range(len(ids))}
            self.is_loaded = True
            print(f"→ Fallback index built with {len(ids)} vectors.")
        
        except Exception:
            print("❌ Error building fallback index.")
            traceback.print_exc()
        
        finally:
            cursor.close()
            conn.close()
    
    def search(self, query_embedding: np.ndarray, k: int = 100):
        """Search the index; lazy‑load it if needed."""
        if not self.is_loaded:
            self.load_index()
        if self.index is None or not self.id_mapping:
            return np.empty((1,0)), np.empty((1,0), dtype=int)
        return self.index.search(query_embedding, k=k)
