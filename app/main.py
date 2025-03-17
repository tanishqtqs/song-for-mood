from fastapi import FastAPI
from pydantic import BaseModel
from app.models.emotion_detector import detect_emotion
from app.models.song_recommender import retrieve_song
import uvicorn

app = FastAPI()

class UserInput(BaseModel):
    text: str

@app.post("/recommend")
def recommend_song(user_input: UserInput):
    mood = detect_emotion(user_input.text)
    song = retrieve_song(mood)
    return {"mood": mood, "recommendation": song}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)