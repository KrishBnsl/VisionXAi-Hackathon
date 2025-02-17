import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

# --------------------------------------
# Load BERT model
# --------------------------------------
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_bert_embeddings(texts):
    return np.array([bert_model.encode(text, convert_to_tensor=True).cpu().numpy() for text in texts])

def train_word2vec(corpus):
    tokenized_corpus = [str(text).split() for text in corpus]
    w2v_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return w2v_model

def preprocess_data(df):
    df['bert_embedding'] = list(generate_bert_embeddings(df['course_title'].astype(str)))
    
    w2v_model = train_word2vec(df['course_title'].astype(str))
    df['w2v_embedding'] = df['course_title'].apply(
        lambda x: np.mean([w2v_model.wv[word] for word in str(x).split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
    )
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['course_title'].astype(str))
    
    svd = TruncatedSVD(n_components=384)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    df['tfidf_vector'] = list(tfidf_reduced)
    
    return df, w2v_model, tfidf_vectorizer

def recommend_courses(df, prev_education, future_goals, difficulty=None, cert_type=None, w2v_model=None):
    user_embedding = bert_model.encode(prev_education + ' ' + future_goals, convert_to_tensor=True).cpu().numpy()
    user_w2v_embedding = np.mean(
        [w2v_model.wv[word] for word in (prev_education + ' ' + future_goals).split() if word in w2v_model.wv] 
        or [np.zeros(100)], axis=0
    )
    
    bert_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['bert_embedding']])
    w2v_similarities = np.array([1 - cosine(user_w2v_embedding, emb) for emb in df['w2v_embedding']])
    tfidf_similarities = np.array([1 - cosine(user_embedding, emb) for emb in df['tfidf_vector']])
    
    final_scores = (0.5 * tfidf_similarities) + (0.3 * bert_similarities) + (0.2 * w2v_similarities)
    df['score'] = final_scores
    
    if difficulty and difficulty != "All":
        df = df[df['course_difficulty'] == difficulty]
    if cert_type and cert_type != "All":
        df = df[df['course_Certificate_type'] == cert_type]
    
    top_courses = df.sort_values(by='score', ascending=False).head(5)
    return top_courses.to_dict(orient='records')

# --------------------------------------
# Load & Preprocess Data
# --------------------------------------
df = pd.read_csv("coursera-data.csv")

df.rename(columns={
    'Name': 'course_title',
    'Url': 'course_url',
    'Rating': 'course_rating',
    'Difficulty': 'course_difficulty'
}, inplace=True)

df['course_rating'] = pd.to_numeric(df['course_rating'], errors='coerce')
df['course_organization'] = "Unknown"
df['course_Certificate_type'] = "Unknown"
df['course_students_enrolled'] = 0

df, w2v_model, _ = preprocess_data(df)

# --------------------------------------
# Custom Cyberpunk CSS
# --------------------------------------
st.markdown(
    """
    <style>
    /* Global body styling */
    body {
        background-color: #000000;
        color: #E0E0E0;
        font-family: 'Orbitron', sans-serif;
        margin: 0;
        padding: 0;
    }

    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #00ff41;
        margin-top: 20px;
        margin-bottom: 20px;
        text-shadow: 0 0 10px #00ff41;
    }

    /* Streamlit default widget labels */
    .css-1lcbmhc, .css-12oz5g7, .css-hnxmzp {
        color: #00ff41 !important;
        font-weight: 600;
    }

    /* The text inputs themselves */
    .stTextArea, .stTextInput {
        background-color: #111111;
        color: #E0E0E0;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00ff41;
        color: #000;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.6em 1em;
        box-shadow: 0 0 10px #00ff41;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #000;
        color: #00ff41;
        box-shadow: 0 0 15px #00ff41, 0 0 20px #00ff41;
        cursor: pointer;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
        box-shadow: 0 0 10px #00ff41;
    }
    [data-testid="stSidebar"] h2 {
        color: #00ff41;
        text-shadow: 0 0 5px #00ff41;
    }

    /* Subheader styling for recommended courses */
    .recommendation-title {
        font-size: 2rem;
        color: #00ff41;
        margin-top: 30px;
        margin-bottom: 20px;
        text-shadow: 0 0 10px #00ff41;
    }

    /* Course card styling */
    .course-card {
        background: #111111;
        border: 1px solid #00ff41;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px #00ff41;
    }
    .course-card h4 {
        color: #00ff41;
        margin-bottom: 10px;
        text-shadow: 0 0 5px #00ff41;
    }
    .course-card p {
        margin: 5px 0;
        color: #E0E0E0;
    }
    .course-card a {
        display: inline-block;
        padding: 10px 20px;
        background: #00ff41;
        color: #000;
        font-weight: bold;
        border-radius: 5px;
        text-decoration: none;
        box-shadow: 0 0 5px #00ff41;
        transition: all 0.3s ease;
    }
    .course-card a:hover {
        background: #000;
        color: #00ff41;
        box-shadow: 0 0 10px #00ff41;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------
# App Title
# --------------------------------------
st.markdown('<h1 class="main-title">AI-Powered Course Recommender</h1>', unsafe_allow_html=True)

# --------------------------------------
# Sidebar Filters
# --------------------------------------
st.sidebar.header("Filters")
difficulty = st.sidebar.selectbox("Difficulty", ["All"] + list(df['course_difficulty'].dropna().unique()))
cert_type = st.sidebar.selectbox("Certificate Type", ["All"] + list(df['course_Certificate_type'].dropna().unique()))

# --------------------------------------
# User Inputs
# --------------------------------------
prev_education = st.text_input("Enter Your Education")
future_goals = st.text_input("Your Future Goals")

# --------------------------------------
# Recommendation Logic
# --------------------------------------
if st.button("Get Recommendations"):
    if prev_education and future_goals:
        recommendations = recommend_courses(df, prev_education, future_goals, difficulty, cert_type, w2v_model)
        if recommendations:
            st.markdown('<div class="recommendation-title">🌟 Recommended Courses</div>', unsafe_allow_html=True)
            for idx, course in enumerate(recommendations):
                st.markdown(f"""
                    <div class="course-card">
                        <h4>{idx+1}. {course['course_title']}</h4>
                        <p><b>Organization:</b> {course['course_organization']}</p>
                        <p><b>Certificate Type:</b> {course['course_Certificate_type']}</p>
                        <p><b>Difficulty:</b> {course['course_difficulty']}</p>
                        <p><b>Rating:</b> {course['course_rating']} &nbsp;|&nbsp; <b>Enrolled:</b> {course['course_students_enrolled']}</p>
                        <a href="{course['course_url']}" target="_blank">Access Course</a>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matching courses found. Try different filters!")
    else:
        st.warning("Please enter both previous education and future goals!")

