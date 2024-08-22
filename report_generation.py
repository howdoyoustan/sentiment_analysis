import streamlit as st
import pandas as pd
import unicodedata
import re
from sqlalchemy import create_engine, text
import nltk
from nltk import ngrams
from nltk.corpus import wordnet
from collections import Counter

# Database connection (SQLite example)
engine = create_engine('sqlite:///brand_analysis.db')

# Function to download NLTK resources with error handling
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

# Ensure NLTK resources are available
download_nltk_resources()

# Helper Function to clean and format brand names for table names
def clean_table_name(brand_name):
    """Remove special characters and replace spaces with underscores in brand names for table names."""
    return re.sub(r'\W+', '_', brand_name.lower())

def check_login(username, password):
    """Check login credentials without password hashing."""
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT * FROM brand_credentials WHERE username = :username AND password = :password"),
            {"username": username, "password": password})
        return result.fetchone() is not None

def generate_and_store_brand_credentials(df):
    """Generate and store brand credentials with a hardcoded password."""
    credentials = []
    password = "pass"
    for brand_name in df['name'].unique():
        username = brand_name  # Use brand name directly as the username
        credentials.append((username, password))

    store_credentials_to_db(credentials)
    return credentials, password  # Return password as it's the same for all brands

def store_credentials_to_db(credentials):
    """Store the generated credentials to the database."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS brand_credentials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))

        for username, password in credentials:
            try:
                insert_sql = text("INSERT INTO brand_credentials (username, password) VALUES (:username, :password)")
                connection.execute(insert_sql, {"username": username, "password": password})
            except Exception as e:
                st.error(f"Error inserting credentials for {username}: {str(e)}")

# Cleaning Pipeline
class CleaningPipeline:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

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

# Brand Analysis Function
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def get_themes_with_bigrams(text):
    words = nltk.word_tokenize(text.lower())
    themes = []

    bigrams = [' '.join(bigram) for bigram in ngrams(words, 2)]

    for bigram in bigrams:
        if any(wordnet.synsets(word, pos=get_wordnet_pos(word)) for word in bigram.split()):
            themes.append(bigram)
    return Counter(themes)

def analyze_brand_with_bigrams(df, brand_name):
    brand_df = df[df['name'].str.contains(brand_name, case=False, na=False)]

    def analyze_group(group):
        themes = get_themes_with_bigrams(' '.join(group['cleaned_text']))
        top_themes = themes.most_common(20)

        positive_themes = [theme for theme, count in top_themes if group['polarity'].mean() > 0]
        negative_themes = [theme for theme, count in top_themes if group['polarity'].mean() <= 0]

        return pd.Series({
            'address': group['address'].iloc[0],
            'postal_code': group['postal_code'].iloc[0],
            'business_ratings': round(group['business_ratings'].mean(), 2),
            'csat': round(group['csat'].mean(), 2),
            'nps': round(group['nps'].mean(), 2),
            'polarity': round(group['polarity'].mean(), 2),
            'strengths': ', '.join(positive_themes[:3]),
            'areas_for_improvement': ', '.join(negative_themes[:3])
        })

    brand_insights = brand_df.groupby('business_id').apply(analyze_group).reset_index()

    # Calculate summary statistics
    total_locations_analyzed = len(brand_insights)
    average_business_rating = brand_insights['business_ratings'].mean()
    highest_rated_location = brand_insights.loc[brand_insights['business_ratings'].idxmax(), 'address']
    lowest_rated_location = brand_insights.loc[brand_insights['business_ratings'].idxmin(), 'address']

    all_strengths = ', '.join(brand_insights['strengths'].dropna())
    all_improvements = ', '.join(brand_insights['areas_for_improvement'].dropna())

    common_strengths = ', '.join(sorted(set(all_strengths.split(', ')))[:9])
    common_areas_for_improvement = ', '.join(sorted(set(all_improvements.split(', ')))[:9])

    # Save the report to the database
    report_data = {
        'total_locations_analyzed': total_locations_analyzed,
        'average_business_rating': average_business_rating,
        'highest_rated_location': highest_rated_location,
        'lowest_rated_location': lowest_rated_location,
        'common_strengths': common_strengths,
        'common_areas_for_improvement': common_areas_for_improvement
    }

    # Create a new DataFrame to hold the summary statistics
    summary_df = pd.DataFrame([report_data])

    # Save the summary to the database
    table_name = f"{clean_table_name(brand_name)}_summary"
    summary_df.to_sql(table_name, engine, if_exists='replace', index=False)

    # Save detailed brand insights
    detailed_table_name = f"{clean_table_name(brand_name)}_insights"
    brand_insights.to_sql(detailed_table_name, engine, if_exists='replace', index=False)

    return brand_insights, summary_df

# Streamlit Application
def main():
    st.title('Brand Analysis')

    # Tab layout
    tab1, tab2 = st.tabs(["Report Generation", "Login & View Report"])

    with tab1:
        st.header('Report Generation')
        uploaded_file = st.file_uploader("Choose a CSV file with brand data", type='csv')

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            pipeline = CleaningPipeline()
            df['cleaned_text'] = df['text'].apply(pipeline.clean_text)

            st.header('Analyze and Save Reports')
            for brand_name in df['name'].unique():
                analyze_brand_with_bigrams(df, brand_name)
            st.success("All reports have been generated and saved to the database!")

            st.header('Generate and Display Credentials')
            credentials, password = generate_and_store_brand_credentials(df)
            credentials_df = pd.DataFrame(credentials, columns=['Username', 'Password'])
            credentials_df['Password'] = password
            st.dataframe(credentials_df)

            st.success(f"Credentials stored successfully! All brands have the password: '{password}'")

    with tab2:
        st.header('Login to View Report')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            if check_login(username, password):
                st.success(f"Logged in as {username}")

                table_name = f"{clean_table_name(username)}_insights"
                query = f"SELECT * FROM {table_name}"
                with engine.connect() as connection:
                    df = pd.read_sql(query, connection)

                st.header(f"Report for {username.capitalize()}")
                st.write(df)
                st.bar_chart(df['business_ratings'])

            else:
                st.error("Invalid credentials. Please try again.")

if __name__ == '__main__':
    main()
