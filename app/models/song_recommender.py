import faiss
import numpy as np
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.config import song_data_path, faiss_index_path

df = pd.read_csv(song_data_path)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index(faiss_index_path)

def retrieve_song(mood_keyword: str, top_k: int = 5) -> str:
    mood_embedding = embedding_model.encode([mood_keyword], convert_to_tensor=True)
    mood_embedding_np = np.array(mood_embedding.cpu().numpy())

    _, indices = index.search(mood_embedding_np, top_k)
    
    recommended_songs = df.iloc[indices[0]][['title', 'artist']].values.tolist()
    
    song = random.choice(recommended_songs)
    return f"You should listen to {song[0]} by {song[1]}."
