# ==============================================
# app.py - Hashtag Recommendation Web App (HYBRID)
# ==============================================

from flask import Flask, render_template_string, request
import joblib, re, spacy
from nltk.corpus import stopwords

# ----------------------------------------------
# INITIALIZE
# ----------------------------------------------
app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Load models
print("⏳ Loading models...")
model = joblib.load("models/hashtag_model.pkl")
tfidf = joblib.load("vectorizers/tfidf_vectorizer.pkl")
mlb = joblib.load("models/label_binarizer.pkl")
print("✅ Models loaded.")

# ----------------------------------------------
# HYBRID LOGIC: KEYWORD MAPPING
# ----------------------------------------------
# If the ML model is unsure, we boost these tags
KEYWORD_MAP = {
    "ai": ["ai", "tech", "innovation", "future", "machinelearning"],
    "python": ["python", "coding", "developer", "programming"],
    "code": ["coding", "tech", "software"],
    "gym": ["fitness", "gym", "health", "workout"],
    "run": ["running", "cardio", "fitness"],
    "food": ["foodie", "yummy", "delicious", "cooking"],
    "cook": ["cooking", "chef", "food"],
    "travel": ["travel", "adventure", "explore", "wanderlust"],
    "beach": ["summer", "beach", "ocean"],
    "happy": ["joy", "positivity", "happiness"],
    "sad": ["mood", "emotional"],
    "music": ["music", "concert", "songs"],
    "game": ["gaming", "gamer", "esports"]
}

# ----------------------------------------------
# CLEANING
# ----------------------------------------------
def clean_tweet(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", "", str(text).lower())
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

# ----------------------------------------------
# RECOMMENDATION ENGINE
# ----------------------------------------------
def recommend_hashtags(post_text, top_k=6):
    cleaned = clean_tweet(post_text)
    
    # 1. Machine Learning Prediction
    vector = tfidf.transform([cleaned])
    proba = model.predict_proba(vector)[0]
    
    # Get top ML predictions (even low probability ones)
    top_indices = proba.argsort()[-top_k:][::-1]
    
    # Dictionary to store {hashtag: score}
    # We lower the threshold to 0.01 to capture weak signals from ML
    scores = {mlb.classes_[i]: proba[i] * 100 for i in top_indices if proba[i] > 0.01}
    
    # 2. Rule-Based Boost (The "Hybrid" Fix)
    input_words = set(cleaned.split())
    
    # Check for keyword matches in the input text
    for keyword, tags in KEYWORD_MAP.items():
        if keyword in post_text.lower() or keyword in input_words:
            for tag in tags:
                # If the tag is already predicted, boost its score to 95-99%
                # If not, add it with high confidence
                scores[tag] = 95.0

    # 3. Sort and Format
    # Sort by score descending
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top K
    return sorted_tags[:top_k]

# ----------------------------------------------
# HTML TEMPLATE
# ----------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hashtag Recommendation System</title>
<style>
body {
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f5;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  min-height: 100vh;
  margin: 0;
}
.container {
  background: #fff;
  padding: 40px;
  border-radius: 15px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  width: 600px;
  max-width: 90%;
  text-align: center;
}
h1 { color: #1da1f2; margin-bottom: 20px; }
p.desc { color: #666; margin-bottom: 25px; }
textarea {
  width: 100%; height: 100px; padding: 15px;
  border: 2px solid #e1e8ed; border-radius: 10px;
  resize: none; font-size: 16px; outline: none;
  box-sizing: border-box; transition: border 0.3s;
}
textarea:focus { border-color: #1da1f2; }
button {
  background: #1da1f2; color: white; border: none;
  padding: 12px 30px; border-radius: 30px; font-size: 16px;
  font-weight: bold; cursor: pointer; margin-top: 20px;
  transition: background 0.3s;
}
button:hover { background: #0c85d0; }
.hashtags {
  margin-top: 30px; display: flex; flex-wrap: wrap;
  gap: 10px; justify-content: center;
}
.tag {
  background: #e8f5fd; color: #1da1f2; font-weight: 600;
  border-radius: 20px; padding: 8px 16px;
  font-size: 15px; display: flex; align-items: center;
}
.score { font-size: 0.8em; opacity: 0.7; margin-left: 5px; }
</style>
</head>
<body>
<div class="container">
  <h1>#Hashtag Generator</h1>
  <p class="desc">Enter your social media caption to get AI predictions.</p>
  <form method="POST">
    <textarea name="tweet" placeholder="e.g., Just finished a great workout at the gym!" required>{{ tweet or '' }}</textarea>
    <br>
    <button type="submit">Generate Hashtags</button>
  </form>

  {% if hashtags %}
  <div class="hashtags">
    {% for tag, score in hashtags %}
      <div class="tag">#{{ tag }} <span class="score">{{ "%.0f"|format(score) }}%</span></div>
    {% endfor %}
  </div>
  {% elif tweet %}
  <p style="margin-top:20px; color: #e0245e;">No hashtags found. Try adding more descriptive words!</p>
  {% endif %}
</div>
</body>
</html>
"""

# ----------------------------------------------
# ROUTES
# ----------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    hashtags = []
    tweet = ""
    if request.method == "POST":
        tweet = request.form["tweet"]
        hashtags = recommend_hashtags(tweet)
    return render_template_string(HTML_TEMPLATE, hashtags=hashtags, tweet=tweet)

# ----------------------------------------------
# RUN APP
# ----------------------------------------------
if __name__ == "__main__":
    print("✅ Web app ready at http://127.0.0.1:5000")
    app.run(debug=True)