import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import re

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    """Lowercase and remove special characters from text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def generate_bert_embeddings(texts):
    """Generate BERT embeddings for a list of texts."""
    return np.array([bert_model.encode(clean_text(text), convert_to_tensor=True).cpu().numpy() for text in texts])

def train_word2vec(corpus):
    """Train a Word2Vec model on the provided corpus."""
    tokenized_corpus = [clean_text(text).split() for text in corpus]
    w2v_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return w2v_model

def preprocess_data(df):
    """Generate embeddings for the dataset."""
    df.dropna(subset=['course_title', 'course_difficulty', 'course_rating'], inplace=True)
    df['clean_title'] = df['course_title'].apply(clean_text)
    
    df['bert_embedding'] = list(generate_bert_embeddings(df['clean_title']))
    
    w2v_model = train_word2vec(df['clean_title'])
    df['w2v_embedding'] = df['clean_title'].apply(
        lambda x: np.mean([w2v_model.wv[word] for word in x.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
    )
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_title'])
    svd = TruncatedSVD(n_components=100)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    df['tfidf_vector'] = list(tfidf_reduced)
    
    return df, w2v_model, tfidf_vectorizer

def recommend_courses(df, prev_education, future_goals, difficulty=None, cert_type=None):
    """Recommend courses based on user input and course similarities."""
    user_text = clean_text(prev_education + ' ' + future_goals)
    user_embedding = bert_model.encode(user_text, convert_to_tensor=True).cpu().numpy()
    
    w2v_model = train_word2vec(df['clean_title'])
    user_w2v_embedding = np.mean(
        [w2v_model.wv[word] for word in user_text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0
    )
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_title'])
    svd = TruncatedSVD(n_components=100)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    user_tfidf_embedding = svd.transform(tfidf_vectorizer.transform([user_text]))[0]
    
    # Compute similarities
    bert_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['bert_embedding']])
    w2v_similarities = np.array([1 - cosine(user_w2v_embedding, emb) for emb in df['w2v_embedding']])
    tfidf_similarities = np.array([1 - cosine(user_tfidf_embedding, emb) for emb in df['tfidf_vector']])
    
    # Final score (weighted combination)
    final_scores = (0.4 * tfidf_similarities) + (0.4 * bert_similarities) + (0.2 * w2v_similarities)
    df['score'] = final_scores
    
    # Filtering
    if difficulty and difficulty != "All":
        df = df[df['course_difficulty'] == difficulty]
    if cert_type and cert_type != "All":
        df = df[df['course_Certificate_type'] == cert_type]
    
    # Sort and return top results
    top_courses = df.sort_values(by='score', ascending=False).head(5)
    return top_courses.to_dict(orient='records')

# Load the updated dataset
df = pd.read_csv('coursera-data.csv')
df, w2v_model, tfidf_vectorizer = preprocess_data(df)