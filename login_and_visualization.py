import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import hashlib
import re

# Database connection
engine = create_engine('sqlite:///brand_analysis.db')

# Sanitize table names by removing special characters and replacing spaces with underscores
def sanitize_table_name(name):
    return re.sub(r'\W+', '_', name.lower())

# Check login credentials
def check_login(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM brand_credentials WHERE username = :username AND password_hash = :password_hash"),
                                    {"username": username, "password_hash": password_hash})
        return result.fetchone() is not None

# Main function for Page 2
def main_page_2():
    st.title('Brand Analysis Login')

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.header('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Logged in as {username}")

                # Automatically display the report after login
                table_name = sanitize_table_name(f"{st.session_state.username}_analysis")
                query = f"SELECT * FROM {table_name}"
                with engine.connect() as connection:
                    df = pd.read_sql(query, connection)

                st.header(f"Report for {st.session_state.username.capitalize()}")
                st.write(df)
                st.bar_chart(df['business_ratings'])

            else:
                st.error("Invalid credentials")
        else:
            st.header(f'Welcome, {st.session_state.username.capitalize()}')
            st.header('Visualize Brand Analysis')

            # Automatically display the report after login
            table_name = sanitize_table_name(f"{st.session_state.username}_analysis")
            query = f"SELECT * FROM {table_name}"
            with engine.connect() as connection:
                df = pd.read_sql(query, connection)

            st.write(df)
            st.bar_chart(df['business_ratings'])

        if __name__ == '__main__':
            main_page_2()
