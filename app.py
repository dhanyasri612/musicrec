from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import io
import os

# Flask app
app = Flask(__name__)

# Load BLIP and classifier models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Emotion labels
emotion_labels = ["happy", "sad", "romantic", "energetic", "calm", "suspenseful", "emotional", "devotional", "nostalgic"]

# Spotify setup
SPOTIFY_CLIENT_ID = 'e7f5c65becdf49529d5dde3249368676'
SPOTIFY_CLIENT_SECRET = '9763c5f8f1514c0b9af2c209371f853b'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

def generate_caption(image):
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

def detect_emotion(caption):
    result = classifier(caption, candidate_labels=emotion_labels)
    return result['labels'][0]

def search_spotify(emotion, limit=10):
    emotion_to_query = {
        "happy": "happy tamil songs",
        "sad": "sad tamil songs",
        "romantic": "romantic tamil songs",
        "energetic": "energetic tamil songs",
        "calm": "soothing tamil songs",
        "emotional": "emotional tamil songs",
        "devotional": "tamil devotional songs",
        "nostalgic": "old tamil songs",
        "suspenseful": "intense background tamil"
    }
    query = emotion_to_query.get(emotion.lower(), "tamil songs")
    results = sp.search(q=query, type="track", limit=limit)

    tracks = []
    for item in results['tracks']['items']:
        tracks.append({
            'name': item['name'],
            'artist': item['artists'][0]['name'],
            'id': item['id'],  # Add the track ID here
            'preview_url': item['preview_url'] or ""
        })

    return tracks

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    
    caption = generate_caption(image)
    emotion = detect_emotion(caption)
    tracks = search_spotify(emotion)

    return jsonify(tracks)

if __name__ == "__main__":
    app.run(debug=True)
