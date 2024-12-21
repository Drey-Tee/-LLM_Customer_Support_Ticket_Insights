import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

base_dir = os.path.dirname(os.path.abspath(__file__))
# Load NLTK's English stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove repetitive words (e.g., "productpurchased productpurchased")
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    # Remove generic phrases (e.g., "please assist", "im issue")
    generic_phrases = ["please assist", "im issue", "note suggest", "kindly"]
    for phrase in generic_phrases:
        text = text.replace(phrase, "")
    # Remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)


def preprocess_dataset(file_path, output_path):
    """Load a dataset, preprocess the text, and clean the data."""
    try:
        df = pd.read_csv(file_path)
        print("DataFrame loaded successfully!")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print("Initial DataFrame:")
    print(df.head())

    # Clean and preprocess text
    if 'Ticket Description' not in df.columns:
        print("Error: 'Ticket Description' column is missing.")
        return

    # Clean and preprocess `Ticket Description`
    df['Cleaned Text'] = df['Ticket Description'].apply(clean_text).apply(preprocess_text)

    # Handle missing values
    df['Resolution'] = df['Resolution'].fillna('No Resolution')
    df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'].fillna(-1)
    df['First Response Time'] = df['First Response Time'].fillna('Not Available')
    df['Time to Resolution'] = df['Time to Resolution'].fillna('Not Available')

    # Convert Date Columns
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')

    # Standardize Categorical Columns
    df['Ticket Priority'] = df['Ticket Priority'].str.lower().str.strip()
    df['Ticket Status'] = df['Ticket Status'].str.lower().str.strip()

    # Add 'Ticket Priority Numeric'
    priority_mapping = {'low': 1, 'medium': 2, 'high': 3}
    df['Ticket Priority Numeric'] = df['Ticket Priority'].map(priority_mapping).fillna(0)

    # Add 'Sentiment Score'
    from textblob import TextBlob

    def get_sentiment_score(text):
        if pd.isna(text):
            return 0  # Neutral sentiment
        return TextBlob(text).sentiment.polarity

    df['Sentiment Score'] = df['Cleaned Text'].apply(get_sentiment_score)
    df['Sentiment Score'] = df['Sentiment Score'].fillna(0)  # Assign neutral sentiment

    # Drop Irrelevant Columns
    if 'Customer Email' in df.columns:
        df = df.drop(columns=['Customer Email'])

    # Final check
    print("Columns after preprocessing:")
    print(df.columns)

    try:
        df.to_csv(output_path, index=False)
        print(f"Preprocessed dataset saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


# Run preprocessing
preprocess_dataset(
    file_path='data/customer_support_tickets.csv',
    output_path='data/cleaned_dataset.csv'

)

