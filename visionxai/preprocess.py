import pandas as pd
import re

def clean_text(text):
    """Lowercase and remove special characters from text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

def preprocess_data(filepath):
    """Load and preprocess course data"""
    df = pd.read_csv(filepath)
    df.rename(columns={
        'Name': 'course_title',
        'Url': 'course_url',
        'Rating': 'course_rating',
        'Difficulty': 'course_difficulty'
    }, inplace=True)
    
    df.dropna(subset=['course_title', 'course_difficulty'], inplace=True)
    df['course_rating'] = pd.to_numeric(df['course_rating'], errors='coerce')
    df['course_organization'] = "Unknown"
    df['course_Certificate_type'] = "Unknown"
    df['course_students_enrolled'] = 0
    
    df['combined_text'] = (
        df['course_title'].fillna('') + " " +
        df['course_organization'].fillna('') + " " +
        df['course_Certificate_type'].fillna('') + " " +
        df['course_difficulty'].fillna('')
    ).apply(clean_text)
    
    return df
