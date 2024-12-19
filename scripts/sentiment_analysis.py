
import pandas as pd
from textblob import TextBlob


def analyze_sentiment(text):
    """Analyze sentiment of given text."""
    if pd.isna(text):  # Handle missing or NaN text entries
        return 0  # Neutral sentiment for missing text
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# Load the preprocessed dataset (output from text_preprocessing.py)
df = pd.read_csv('/Users/dreytee/PycharmProjects/LLM_Ticket_Insights/data/cleaned_dataset.csv')

# Apply sentiment analysis on a relevant text column (e.g., 'Cleaned Text')
if 'Cleaned Text' in df.columns:
    df['Sentiment Score'] = df['Cleaned Text'].apply(analyze_sentiment)
else:
    raise KeyError("'Cleaned Text' column not found in the dataset. Make sure preprocessing is complete.")

# Save the updated DataFrame with Sentiment Score
df.to_csv('/Users/dreytee/PycharmProjects/LLM_Ticket_Insights/data/sentiment_dataset.csv', index=False)

# Verification
print("Sentiment analysis completed and saved successfully.")
print(df.head())
