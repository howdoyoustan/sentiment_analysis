import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# Database connection
engine = create_engine('sqlite:///brand_analysis.db')


# CleaningPipeline class (as before)
class CleaningPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def remove_non_ascii(self, text):
        return ''.join(c for c in text if ord(c) < 128)

    def remove_whitespace(self, text):
        return ' '.join(text.split())

    def remove_special_characters(self, text):
        return re.sub(r'[^a-zA-Z0-9\\s]', '', text)

    def remove_stopwords(self, text):
        return ' '.join(word for word in text.split() if word.lower() not in self.stop_words)

    def lemmatize_words(self, text):
        return ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())

    def clean_text(self, text):
        text = self.remove_non_ascii(text)
        text = self.remove_whitespace(text)
        text = self.remove_special_characters(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_words(text)
        return text

# Function to sanitize table names
def sanitize_table_name(name):
    return re.sub(r'\\W+', '_', name.lower())


# Function to extract bigrams
def extract_bigrams(text):
    words = nltk.word_tokenize(text.lower())
    return list(ngrams(words, 2))


# Function to analyze sentiment of bigrams
def analyze_bigram_sentiment(bigrams, text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return [(bigram, sentiment) for bigram in bigrams]


# Function to get top themes
def get_top_themes(bigram_sentiments, top_n=10):
    positive_themes = Counter()
    negative_themes = Counter()

    for bigram, sentiment in bigram_sentiments:
        if sentiment > 0:
            positive_themes[bigram] += 1
        elif sentiment < 0:
            negative_themes[bigram] += 1

    return positive_themes.most_common(top_n), negative_themes.most_common(top_n)


# Function to analyze brand with bigrams
def analyze_brand_with_bigrams(df, brand_name):
    brand_data = df[df['name'] == brand_name]

    all_bigrams = []
    all_bigram_sentiments = []

    for _, row in brand_data.iterrows():
        bigrams = extract_bigrams(row['clean_text'])
        all_bigrams.extend(bigrams)
        all_bigram_sentiments.extend(analyze_bigram_sentiment(bigrams, row['text']))

    positive_themes, negative_themes = get_top_themes(all_bigram_sentiments)

    analysis = {
        'brand_name': brand_name,
        'total_locations': len(brand_data),
        'average_business_rating': brand_data['business_ratings'].mean(),
        'average_user_rating': brand_data['user_ratings'].mean(),
        'total_reviews': brand_data['review_count'].sum(),
        'average_sentiment_score': brand_data['sentiment_score'].mean(),
        'average_nps': brand_data['nps'].mean(),
        'average_csat': brand_data['csat'].mean(),
        'positive_themes': ', '.join(positive_themes),
        'negative_themes': ', '.join(negative_themes),
    }

    return pd.DataFrame([analysis]), brand_data, all_bigram_sentiments


# Function to store brand report
def store_brand_report(brand_name, report_df, brand_data, bigram_sentiments):
    table_name = sanitize_table_name(f"{brand_name}_report")
    report_df.to_sql(table_name, engine, if_exists='replace', index=False)

    data_table_name = sanitize_table_name(f"{brand_name}_data")
    brand_data.to_sql(data_table_name, engine, if_exists='replace', index=False)

    bigram_sentiments_df = pd.DataFrame(bigram_sentiments, columns=['bigram', 'sentiment'])
    bigram_table_name = sanitize_table_name(f"{brand_name}_bigrams")
    bigram_sentiments_df.to_sql(bigram_table_name, engine, if_exists='replace', index=False)


# Function to generate and store credentials
def generate_and_store_credentials(brand_names):
    credentials = []
    for brand_name in brand_names:
        username = sanitize_table_name(brand_name)
        password = "pass"  # Hardcoded password
        credentials.append((username, password, brand_name))

    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS brand_credentials (
            username TEXT PRIMARY KEY,
            password TEXT,
            brand_name TEXT
        )
        """))
        for username, password, brand_name in credentials:
            conn.execute(text("""
            INSERT OR REPLACE INTO brand_credentials (username, password, brand_name)
            VALUES (:username, :password, :brand_name)
            """), {'username': username, 'password': password, 'brand_name': brand_name})
        conn.commit()

# Function to check login
def check_login(username, password):
    with engine.connect() as conn:
        result = conn.execute(text("""
        SELECT * FROM brand_credentials 
        WHERE username = :username AND password = :password
        """), {"username": username, "password": password})
        return result.fetchone()



# Function to create word cloud
def create_word_cloud(themes):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(themes))
    return wordcloud


# Main function for Streamlit app
def main():
    st.title('Brand Analysis App')

    tab1, tab2 = st.tabs(["Upload & Analyze", "Login & View Report"])

    with tab1:
        st.header('Upload CSV and Generate Reports')
        uploaded_file = st.file_uploader("Choose a CSV file with brand data", type='csv')

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            pipeline = CleaningPipeline()
            df['clean_text'] = df['text'].apply(pipeline.clean_text)
            brand_names = df['name'].unique()

            for brand_name in brand_names:
                report, brand_data, bigram_sentiments = analyze_brand_with_bigrams(df, brand_name)
                store_brand_report(brand_name, report, brand_data, bigram_sentiments)

            generate_and_store_credentials(brand_names)

            st.success("All reports have been generated and credentials stored!")

            # Display credentials
            with engine.connect() as conn:
                credentials_df = pd.read_sql("SELECT username, password, brand_name FROM brand_credentials", conn)
            st.dataframe(credentials_df)

    with tab2:
        st.header('Login to View Report')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            user = check_login(username, password)
            if user:
                st.success(f"Logged in as {user.brand_name}")

                report_table_name = sanitize_table_name(f"{user.brand_name}_report")
                data_table_name = sanitize_table_name(f"{user.brand_name}_data")
                bigram_table_name = sanitize_table_name(f"{user.brand_name}_bigrams")

                with engine.connect() as conn:
                    report_df = pd.read_sql(f"SELECT * FROM {report_table_name}", conn)
                    brand_data = pd.read_sql(f"SELECT * FROM {data_table_name}", conn)
                    bigram_sentiments = pd.read_sql(f"SELECT * FROM {bigram_table_name}", conn)

                st.header(f"Report for {user.brand_name}")
                st.write(report_df)

                # Visualizations
                st.subheader("Visualizations")

                # 1. Bar Chart of Average Ratings
                fig, ax = plt.subplots()
                ratings = ['Business Rating', 'User Rating']
                averages = [report_df['average_business_rating'].values[0], report_df['average_user_rating'].values[0]]
                ax.bar(ratings, averages)
                ax.set_ylabel('Average Rating')
                ax.set_title(f'Average Ratings for {user.brand_name}')
                st.pyplot(fig)

                # 2. Sentiment Distribution
                fig, ax = plt.subplots()
                sns.histplot(brand_data['sentiment_score'], kde=True, ax=ax)
                ax.set_xlabel('Sentiment Score')
                ax.set_title(f'Sentiment Distribution for {user.brand_name}')
                st.pyplot(fig)

                # 3. NPS Distribution
                fig, ax = plt.subplots()
                sns.histplot(brand_data['nps'], kde=True, ax=ax)
                ax.set_xlabel('NPS Score')
                ax.set_title(f'NPS Distribution for {user.brand_name}')
                st.pyplot(fig)

                # 4. Scatter Plot of Check-ins vs. User Ratings
                fig, ax = plt.subplots()
                ax.scatter(brand_data['checkins'], brand_data['user_ratings'])
                ax.set_xlabel('Check-ins')
                ax.set_ylabel('User Ratings')
                ax.set_title(f'Check-ins vs. User Ratings for {user.brand_name}')
                st.pyplot(fig)

                # 5. Geographical Distribution
                st.subheader("Geographical Distribution")
                st.map(brand_data[['latitude', 'longitude']])

                # 6. Word Clouds for Positive and Negative Themes
                st.subheader("Themes Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Positive Themes")
                    positive_wordcloud = create_word_cloud(report_df['positive_themes'].values[0])
                    fig, ax = plt.subplots()
                    ax.imshow(positive_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.write("Negative Themes")
                    negative_wordcloud = create_word_cloud(report_df['negative_themes'].values[0])
                    fig, ax = plt.subplots()
                    ax.imshow(negative_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                # 7. Common Strengths and Areas for Improvement
                st.subheader("Common Strengths")
                st.write(", ".join(report_df['positive_themes'].values[0][:5]))

                st.subheader("Areas for Improvement")
                st.write(", ".join(report_df['negative_themes'].values[0][:5]))

            else:
                st.error("Invalid credentials. Please try again.")


if __name__ == '__main__':
    main()
