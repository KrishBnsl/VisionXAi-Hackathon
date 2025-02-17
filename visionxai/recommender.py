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

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_bert_embeddings(texts):
    """Generate BERT embeddings for a list of texts."""
    return np.array([bert_model.encode(text, convert_to_tensor=True).cpu().numpy() for text in texts])

def train_word2vec(corpus):
    """Train a Word2Vec model on the provided corpus."""
    tokenized_corpus = [text.split() for text in corpus]
    w2v_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return w2v_model

def preprocess_data(df):
    """Generate embeddings for the dataset."""
    df['bert_embedding'] = list(generate_bert_embeddings(df['course_title']))
    
    w2v_model = train_word2vec(df['course_title'])
    df['w2v_embedding'] = df['course_title'].apply(
        lambda x: np.mean([w2v_model.wv[word] for word in x.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
    )
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['course_title'])
    df['tfidf_vector'] = list(tfidf_matrix.toarray())
    
    return df, w2v_model, tfidf_vectorizer

def train_collaborative_model(df):
    """Train a collaborative filtering model using SVD."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['course_title', 'course_rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model

def recommend_courses(df, prev_education, future_goals, difficulty=None, cert_type=None):
    """Recommend courses based on user input and course similarities."""
    user_embedding = bert_model.encode(prev_education + ' ' + future_goals, convert_to_tensor=True).cpu().numpy()
    
    # Compute similarities
    bert_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['bert_embedding']])
    w2v_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['w2v_embedding']])
    tfidf_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['tfidf_vector']])
    
    # Final score (weighted combination)
    final_scores = (0.5 * tfidf_similarities) + (0.3 * bert_similarities) + (0.2 * w2v_similarities)
    df['score'] = final_scores
    
    # Filtering
    if difficulty and difficulty != "All":
        df = df[df['course_difficulty'] == difficulty]
    if cert_type and cert_type != "All":
        df = df[df['course_Certificate_type'] == cert_type]
    
    # Sort and return top results
    top_courses = df.sort_values(by='score', ascending=False).head(5)
    return top_courses.to_dict(orient='records')
