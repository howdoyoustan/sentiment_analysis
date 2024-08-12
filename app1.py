import nltk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import create_engine, text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import re
import unicodedata
import os

# Download necessary NLTK data (only stopwords now)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


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


def load_pickle_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"Model file '{file_path}' not found. Please ensure the model file is available.")
        return None


def process_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep='\	')
    else:
        st.error("Unsupported file format. Please upload a CSV or TXT file.")
        return None

    pipeline = CleaningPipeline()
    df['cleaned_text'] = df['text'].apply(pipeline.clean_text)
    return df


def make_predictions(model, df):
    if model is None:
        st.error("No model loaded. Unable to make predictions.")
        return df
    predictions = model.predict(df['cleaned_text'])
    df['prediction'] = predictions
    df['sentiment'] = df['prediction'].map({0: 'Negative', 1: 'Positive'})
    return df


def visualize_predictions(df):
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)


def create_wordcloud(df, sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud for {sentiment} Sentiment')
    st.pyplot(fig)


def store_predictions(df):
    engine = create_engine('sqlite:///predictions.db')

    # Define the SQL statement for creating the table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        cleaned_text TEXT,
        prediction INTEGER,
        sentiment TEXT
    )
    """

    # Execute the SQL statement using the engine
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))

    # Store the predictions in the database
    df.to_sql('predictions', engine, if_exists='append', index=False)


def main():
    st.title('Sentiment Analysis App')

    # Load the pre-trained model directly
    model = load_pickle_file('SGD_Classifier.pkl')

    # Create tabs
    tab1, tab2 = st.tabs(["Upload & Predict", "Save & Visualize"])

    with tab1:
        st.header('Upload File and Make Predictions')
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'])

        if uploaded_file is not None:
            df = process_file(uploaded_file)
            if df is not None and model is not None:
                if st.button('Make Predictions'):
                    df = make_predictions(model, df)
                    st.write(df)
                    st.session_state['predictions_df'] = df
            elif model is None:
                st.warning("Model file not found. Please ensure the model file is available.")

    with tab2:
        st.header('Save Predictions and Visualize Results')
        if 'predictions_df' in st.session_state:
            df = st.session_state['predictions_df']
            if st.button('Store Predictions'):
                store_predictions(df)
                st.success('Predictions stored successfully!')

            visualize_predictions(df)
            create_wordcloud(df, 'Positive')
            create_wordcloud(df, 'Negative')
        else:
            st.warning('Please upload a file and make predictions first.')


if __name__ == '__main__':
    main()

print("Streamlit code has been updated and is ready to run.")