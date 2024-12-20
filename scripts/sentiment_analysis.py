import pandas as pd
from textblob import TextBlob
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


def analyze_sentiment(text):
    """Analyze sentiment of given text."""
    if pd.isna(text):  # Handle missing or NaN text entries
        return 0  # Neutral sentiment for missing text
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# Load the preprocessed dataset (output from text_preprocessing.py)
# Define file paths relative to the base directory
cleaned_dataset = os.path.join(base_dir, 'data', 'cleaned_dataset.csv')
sentiment_dataset = os.path.join(base_dir, 'data', 'sentiment_dataset.csv')
df = pd.read_csv(cleaned_dataset)

# Apply sentiment analysis on a relevant text column (e.g., 'Cleaned Text')
if 'Cleaned Text' in df.columns:
    df['Sentiment Score'] = df['Cleaned Text'].apply(analyze_sentiment)
else:
    raise KeyError("'Cleaned Text' column not found in the dataset. Make sure preprocessing is complete.")

# Save the updated DataFrame with Sentiment Score
df.to_csv(sentiment_dataset, index=False)

# Verification
print("Sentiment analysis completed and saved successfully.")
print(df.head())
