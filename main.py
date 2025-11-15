# ==============================================
# main.py - Model Training Script (Fixed for CSV Errors)
# ==============================================

import pandas as pd
import re
import nltk
import spacy
import joblib
import os
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------------------------------
# STEP 1: INITIAL SETUP
# ----------------------------------------------
print("🚀 Initializing environment...")

# Download NLTK data if missing
nltk.download('stopwords', quiet=True)

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Spacy model not found. Downloading now...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

# ----------------------------------------------
# STEP 2: LOAD DATASET (ROBUST LOADING)
# ----------------------------------------------
print("\n📂 Loading dataset...")

file_path = 'dataset/twitter_data.csv'
if not os.path.exists(file_path):
    file_path = 'twitter_data.csv'  # Check root directory if not in folder

try:
    # on_bad_lines='skip' tells Pandas to ignore rows with extra commas that cause crashes
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    except TypeError:
        # Fallback for older versions of Pandas
        df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
        
    print(f"✅ CSV loaded successfully from {file_path}")

except Exception as e:
    print(f"❌ Critical Error loading CSV: {e}")
    exit()

# Basic cleanup
df = df[['tweet_text', 'hashtags']].dropna()
print(f"📊 Total valid posts loaded: {len(df)}")

# ----------------------------------------------
# STEP 3: CLEANING TEXT
# ----------------------------------------------
def clean_tweet(text):
    # Remove URLs, Mentions, Hashtags, Special Chars
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", "", str(text).lower())
    doc = nlp(text)
    # Lemmatize and remove stopwords
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

print("\n🧹 Cleaning text data (this may take a moment)...")
df['clean_text'] = df['tweet_text'].apply(clean_tweet)

# ----------------------------------------------
# STEP 4: PROCESS HASHTAGS
# ----------------------------------------------
def process_hashtags(tag_str):
    # Clean hashtags: remove '#', split by commas/spaces
    tags = [t.strip().lower().replace('#', '') for t in str(tag_str).replace(',', ' ').split() if t.strip()]
    return tags

df['hashtags'] = df['hashtags'].apply(process_hashtags)

# --- Count Frequencies ---
all_hashtags = [tag for tags in df['hashtags'] for tag in tags]
hashtag_counts = Counter(all_hashtags)

print(f"📊 Top 5 Hashtags: {hashtag_counts.most_common(5)}")

# --- Filter Logic (Keep > 0 to allow rare tags for small datasets) ---
df['hashtags'] = df['hashtags'].apply(lambda tags: [t for t in tags if hashtag_counts[t] > 0])
df = df[df['hashtags'].map(len) > 0]  # Remove posts with 0 hashtags

print(f"✅ Final Training Count: {len(df)} posts")

# ----------------------------------------------
# STEP 5: FEATURE EXTRACTION
# ----------------------------------------------
print("\n🧠 Vectorizing text...")
# max_features limits the vocabulary size to prevent overfitting on small data
tfidf = TfidfVectorizer(max_features=5000) 
X = tfidf.fit_transform(df['clean_text'])

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['hashtags'])
print(f"🏷️  Model will learn {len(mlb.classes_)} unique hashtags.")

# ----------------------------------------------
# STEP 6: TRAIN/TEST SPLIT
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ----------------------------------------------
# STEP 7: MODEL TRAINING
# ----------------------------------------------
print("\n⚙️  Training Random Forest Model...")
# n_jobs=-1 uses all CPU cores for faster training
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model = OneVsRestClassifier(rf)
model.fit(X_train, y_train)

# ----------------------------------------------
# STEP 8: SAVE MODELS
# ----------------------------------------------
# Create directories if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('vectorizers'):
    os.makedirs('vectorizers')

print("\n💾 Saving artifacts...")
joblib.dump(model, 'models/hashtag_model.pkl')
joblib.dump(tfidf, 'vectorizers/tfidf_vectorizer.pkl')
joblib.dump(mlb, 'models/label_binarizer.pkl')
print("✅ Models saved successfully.")

# ----------------------------------------------
# STEP 9: EVALUATION
# ----------------------------------------------
print("\n📈 Evaluating Performance...")
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

print("-" * 30)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("-" * 30)

print("\n🎉 Done! You can now run 'python app.py'")