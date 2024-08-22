import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import hashlib
import secrets
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import unicodedata
from collections import Counter
import re

# Database connection
engine = create_engine('sqlite:///brand_analysis.db')

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Sanitize table names by removing special characters and replacing spaces with underscores
def sanitize_table_name(name):
    return re.sub(r'\W+', '_', name.lower())

# Generate and Store Brand Credentials
def generate_brand_credentials(df):
    credentials = []
    credentials_display = []
    for brand_name in df['name'].unique():
        username = sanitize_table_name(brand_name)
        password = secrets.token_hex(8)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        credentials.append((username, password_hash))
        credentials_display.append((username, password))
    return credentials, credentials_display

def store_credentials_to_db(credentials):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS brand_credentials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT
    )
    """
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))

    with engine.connect() as connection:
        for username, password_hash in credentials:
            insert_sql = text("INSERT INTO brand_credentials (username, password_hash) VALUES (:username, :password_hash)")
            connection.execute(insert_sql, {"username": username, "password_hash": password_hash})

# Cleaning Pipeline and Brand Analysis
class CleaningPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def remove_non_ascii(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def remove_whitespace(self, text):
        return ' '.join(text.split())

    def remove_special_characters(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def remove_stopwords(self, text):
        return ' '.join([word for word in text.split() if word.lower() not in self.stop_words])

    def lemmatize_words(self, text):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def clean_text(self, text):
        text = self.remove_non_ascii(text)
        text = self.remove_whitespace(text)
        text = self.remove_special_characters(text)
        text = text.lower()
        text = self.remove_stopwords(text)
        text = self.lemmatize_words(text)
        return text

def get_themes_with_bigrams(text):
    words = text.lower().split()
    themes = []

    bigrams = [' '.join(bigram) for bigram in ngrams(words, 2)]

    for bigram in bigrams:
        if all(wordnet.synsets(word, pos=wordnet.NOUN) for word in bigram.split()):
            themes.append(bigram)
    return Counter(themes)

def analyze_brand(df, brand_name):
    brand_df = df[df['name'].str.contains(brand_name, case=False, na=False)]

    def analyze_group(group):
        themes = get_themes_with_bigrams(' '.join(group['cleaned_text']))
        top_themes = themes.most_common(20)

        positive_themes = [theme for theme, count in top_themes if group['polarity'].mean() > 0]
        negative_themes = [theme for theme, count in top_themes if group['polarity'].mean() <= 0]

        return {
            'business_id': group['business_id'].iloc[0],
            'address': group['address'].iloc[0],
            'postal_code': group['postal_code'].iloc[0],
            'business_ratings': round(group['business_ratings'].mean(), 2),
            'csat': round(group['csat'].mean(), 2),
            'nps': round(group['nps'].mean(), 2),
            'polarity': round(group['polarity'].mean(), 2),
            'strengths': ', '.join(positive_themes[:3]),
            'areas_for_improvement': ', '.join(negative_themes[:3])
        }

    brand_insights = brand_df.groupby('business_id').apply(analyze_group).to_dict()
    return brand_insights

def store_analysis_to_db(brand_analysis, brand_name):
    table_name = sanitize_table_name(f"{brand_name}_analysis")
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        business_id TEXT,
        address TEXT,
        postal_code TEXT,
        business_ratings REAL,
        csat REAL,
        nps REAL,
        polarity REAL,
        strengths TEXT,
        areas_for_improvement TEXT,
        total_locations_analyzed INTEGER,
        average_business_rating REAL,
        highest_rated_location TEXT,
        lowest_rated_location TEXT,
        common_strengths TEXT,
        common_areas_for_improvement TEXT
    )
    """

    with engine.connect() as connection:
        connection.execute(text(create_table_sql))

    analysis_df = pd.DataFrame.from_dict(brand_analysis, orient='index')
    analysis_df.reset_index(drop=True, inplace=True)
    analysis_df.to_sql(table_name, engine, if_exists='append', index=False)

# Main function for Page 1
def main_page_1():
    st.title('Brand Analysis Report Generation')

    st.header('Step 1: Upload CSV and Generate Reports')
    uploaded_file = st.file_uploader("Choose a CSV file with brand data", type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        pipeline = CleaningPipeline()
        df['cleaned_text'] = df['text'].apply(pipeline.clean_text)

        st.header('Step 2: Analyze and Save Reports')
        for brand_name in df['name'].unique():
            brand_analysis = analyze_brand(df, brand_name)
            store_analysis_to_db(brand_analysis, brand_name)
        st.success("All reports have been generated and saved to the database!")

        st.header('Step 3: Generate and Display Credentials')
        credentials, credentials_display = generate_brand_credentials(df)
        credentials_df = pd.DataFrame(credentials_display, columns=['Username', 'Password'])
        st.dataframe(credentials_df)

        if st.button('Store Credentials'):
            store_credentials_to_db(credentials)
            st.session_state.credentials_generated = True
            st.success("Credentials stored successfully!")
            st.session_state.ready_for_login = True

        if st.session_state.get('ready_for_login'):
            st.header('Proceed to Login')
            if st.button('Go to Login Page'):
                st.session_state.page = "login"

if __name__ == '__main__':
    if 'page' not in st.session_state:
        st.session_state.page = 'report_generation'
    if st.session_state.page == 'report_generation':
        main_page_1()
    elif st.session_state.page == 'login':
        import login_and_visualization
        login_and_visualization.main_page_2()
