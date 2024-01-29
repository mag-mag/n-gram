import streamlit as st
import requests
from nltk import ngrams
import trafilatura
from nltk.corpus import stopwords
import pandas as pd  # Import pandas for DataFrame creation

# Load Persian stop words
persian_stopwords = pd.read_csv("Persian_stopwords.csv")["stopwords"].to_list()


# Set page title
st.title("URL N-Gram Analyzer")

# Define a function to fetch text from a URL, removing stop words
def fetch_text(url):
    try:
        req = requests.get(url=url)
        req.raise_for_status()  # Raise an exception for non-200 status codes
        pure_text = trafilatura.extract(req.text)
        filtered_text = " ".join([word for word in pure_text.split() if word not in persian_stopwords])  # Remove stop words
        return filtered_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching text from {url}: {e}")
        return ""

# Define a function to calculate n-grams from text, including 1-grams
def calculate_ngrams(text, n):
    filtered_words = [word for word in text.split() if word not in persian_stopwords]  # Filter stop words
    ngrams_list = ngrams(filtered_words, n)
    ngrams_counts = {}
    for ngram in ngrams_list:
        ngram_str = " ".join(ngram)
        ngrams_counts[ngram_str] = ngrams_counts.get(ngram_str, 0) + 1

    return ngrams_counts

# Create input fields for URLs, n-gram value, and frequency filter value
st.header("URL and N-Gram Input")
urls = st.text_area("Enter URLs (one per line)", placeholder="https://example.com\nhttps://another-example.com")
n_gram = st.number_input("Enter n-gram value (1 or more)", min_value=1)
frequency_filter = st.number_input("Enter frequency filter value", min_value=1)

# Process URLs and display results when the "Analyze" button is clicked
if st.button("Analyze"):
    url_list = urls.splitlines()
    df = pd.DataFrame()  # Create an empty DataFrame
    for url in url_list:
        text = fetch_text(url)
        if text:
            ngrams_counts = calculate_ngrams(text, n_gram)

            # Filter n-grams by frequency
            filtered_ngrams = {k: v for k, v in ngrams_counts.items() if v >= frequency_filter}

            ngrams_df = pd.DataFrame.from_dict(filtered_ngrams, orient='index', columns=[url])  # Create DataFrame for each URL
            df = pd.concat([df, ngrams_df], axis=1)  # Combine DataFrames

    if not df.empty:
        st.dataframe(df)  # Display the final DataFrame
    else:
        st.write("No n-grams found.")