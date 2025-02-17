import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
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
    
    # Reduce TF-IDF vector dimension to match BERT (384)
    svd = TruncatedSVD(n_components=384)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    df['tfidf_vector'] = list(tfidf_reduced)
    
    return df, w2v_model, tfidf_vectorizer

def train_collaborative_model(df):
    """Train a collaborative filtering model using SVD."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['course_title', 'course_rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model

def get_w2v_embedding(text, w2v_model):
    """Generate Word2Vec embedding for user input."""
    return np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)

def recommend_courses(df, prev_education, future_goals, difficulty=None, cert_type=None, w2v_model=None):
    """Recommend courses based on user input and course similarities."""
    user_embedding = bert_model.encode(prev_education + ' ' + future_goals, convert_to_tensor=True).cpu().numpy()
    user_w2v_embedding = get_w2v_embedding(prev_education + ' ' + future_goals, w2v_model)
    
    # Compute similarities
    bert_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['bert_embedding']])
    w2v_similarities = np.array([1 - cosine(user_w2v_embedding, emb) for emb in df['w2v_embedding']])
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

# Streamlit UI
st.title("🎓 AI-Powered Course Recommender")

df = pd.read_csv("coursea_data.csv")
df, w2v_model, _ = preprocess_data(df)

st.sidebar.header("🎯 Filters")

difficulty_options = ["All"] + list(df['course_difficulty'].dropna().unique())
cert_type_options = ["All"] + list(df['course_Certificate_type'].dropna().unique())

difficulty = st.sidebar.selectbox("📌 Difficulty", difficulty_options)
cert_type = st.sidebar.selectbox("📜 Certificate Type", cert_type_options)

prev_education = st.text_area("📚 Your Previous Education")
future_goals = st.text_area("🚀 Your Future Goals")

if st.button("🔍 Get Recommendations"):
    if prev_education and future_goals:
        recommendations = recommend_courses(df, prev_education, future_goals, difficulty, cert_type, w2v_model)
        if recommendations:
            st.subheader("🔥 Recommended Courses")
            for idx, course in enumerate(recommendations):
                st.write(f"{idx+1}. {course['course_title']} - {course['course_organization']}")
        else:
            st.warning("⚠️ No courses match your criteria. Try different filters.")
    else:
        st.warning("⚠️ Please enter both previous education and future goals.")
