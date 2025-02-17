import pandas as pd
import re

def clean_text(text):
    """Lowercase and remove special characters from text"""
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

def preprocess_data(filepath):
    """Load and preprocess course data"""
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    # Create a combined text feature for NLP
    df['combined_text'] = (
        df['course_title'] + " " +
        df['course_organization'] + " " +
        df['course_Certificate_type'] + " " +
        df['course_difficulty']
    ).apply(clean_text)

    return df
