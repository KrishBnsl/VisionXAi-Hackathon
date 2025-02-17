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
    return np.array([bert_model.encode(text, convert_to_tensor=True).cpu().numpy() for text in texts])

def train_word2vec(corpus):
    tokenized_corpus = [text.split() for text in corpus]
    w2v_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return w2v_model

def preprocess_data(df):
    df['bert_embedding'] = list(generate_bert_embeddings(df['course_title']))
    
    w2v_model = train_word2vec(df['course_title'])
    df['w2v_embedding'] = df['course_title'].apply(
        lambda x: np.mean([w2v_model.wv[word] for word in x.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
    )
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['course_title'])
    
    svd = TruncatedSVD(n_components=384)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    df['tfidf_vector'] = list(tfidf_reduced)
    
    return df, w2v_model, tfidf_vectorizer

def recommend_courses(df, prev_education, future_goals, difficulty=None, cert_type=None, w2v_model=None):
    user_embedding = bert_model.encode(prev_education + ' ' + future_goals, convert_to_tensor=True).cpu().numpy()
    user_w2v_embedding = np.mean(
        [w2v_model.wv[word] for word in (prev_education + ' ' + future_goals).split() if word in w2v_model.wv] or [np.zeros(100)], axis=0
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

# Sci-Fi UI Design
st.markdown(
    """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        body {
            font-family: 'Orbitron', sans-serif;
            background: url("https://mir-s3-cdn-cf.behance.net/project_modules/hd/33480837562161.57449903490a3.gif") center/cover no-repeat fixed;
            color: #00FF41;
            animation: fadeIn 2s ease-in-out;
        }
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            padding: 20px;
            background: linear-gradient(90deg, #0ff, #00ff41);
            -webkit-background-clip: text;
            color: transparent;
            text-shadow: 0px 0px 10px #0ff;
        }
        .course-card {
            background: rgba(0, 255, 65, 0.1);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px #0ff;
            margin-bottom: 15px;
        }
        .course-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 20px #0ff;
        }
        .btn {
            display: inline-block;
            padding: 8px 15px;
            background: #0ff;
            color: black;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 10px;
        }
        .btn:hover {
            background: #00ff41;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="title">🚀 AI-Powered Course Recommender</div>', unsafe_allow_html=True)

df = pd.read_csv("coursea_data.csv")
df, w2v_model, _ = preprocess_data(df)

st.sidebar.header("🛸 Filters")
difficulty = st.sidebar.selectbox("🧪 Difficulty", ["All"] + list(df['course_difficulty'].dropna().unique()))
cert_type = st.sidebar.selectbox("📜 Certificate Type", ["All"] + list(df['course_Certificate_type'].dropna().unique()))

prev_education = st.text_area("🧠 Enter Your Education")
future_goals = st.text_area("🌌 Your Future Goals")

if st.button("⚡ Get Recommendations"):
    if prev_education and future_goals:
        recommendations = recommend_courses(df, prev_education, future_goals, difficulty, cert_type, w2v_model)
        if recommendations:
            st.subheader("🌟 Recommended Courses")
            for idx, course in enumerate(recommendations):
                st.markdown(f"""
                    <div class="course-card">
                        <h4>{idx+1}. {course['course_title']}</h4>
                        <p>🏫 <b>Organization:</b> {course['course_organization']}</p>
                        <p>📜 <b>Certificate Type:</b> {course['course_Certificate_type']}</p>
                        <p>🎯 <b>Difficulty:</b> {course['course_difficulty']}</p>
                        <p>⭐ <b>Rating:</b> {course['course_rating']} | 👨‍🎓 <b>Enrolled:</b> {course['course_students_enrolled']}</p>
                        <a href="{course.get('course_url', '#')}" class="btn" target="_blank">Access Course</a>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("🚨 No matching courses. Try different filters!")
    else:
        st.warning("🚨 Please enter both previous education and future goals!")
