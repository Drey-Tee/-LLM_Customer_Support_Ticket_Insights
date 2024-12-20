from scripts.text_preprocessing import preprocess_dataset
from scripts.eda import (
    explore_data,
    plot_sentiment_distribution,
    plot_ticket_priority,
    plot_ticket_status,
    plot_customer_satisfaction,
)
from scripts.sentiment_analysis import analyze_sentiment
from scripts.wordcloud_generator import generate_wordcloud
import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
raw_data_path = os.path.join(base_dir, 'data', 'customer_support_tickets.csv')
cleaned_data_path = os.path.join(base_dir, 'data', 'cleaned_dataset.csv')
static_dir = os.path.join(base_dir, 'static')

# Ensure static directory exists
os.makedirs(static_dir, exist_ok=True)

# Static output paths
wordcloud_output_path = os.path.join(static_dir, 'wordcloud_output.png')
sentiment_analysis_output_path = os.path.join(static_dir, 'sentiment_output.png')
priority_output_path = os.path.join(static_dir, 'priority_output.png')
status_output_path = os.path.join(static_dir, 'status_output.png')
satisfaction_output_path = os.path.join(static_dir, 'satisfaction_output.png')

# Step 1: Preprocess Text Data
print("Preprocessing text data...")
preprocess_dataset(raw_data_path, cleaned_data_path)

# Step 2: EDA
print("\nExploring dataset...")
explore_data(cleaned_data_path)

# Step 3: Visualizations
print("\nVisualizing sentiment distribution...")
plot_sentiment_distribution(cleaned_data_path, sentiment_analysis_output_path)

print("\nVisualizing ticket priority distribution...")
plot_ticket_priority(cleaned_data_path, priority_output_path)

print("\nVisualizing ticket status distribution...")
plot_ticket_status(cleaned_data_path, status_output_path)

print("\nVisualizing customer satisfaction ratings...")
plot_customer_satisfaction(cleaned_data_path, satisfaction_output_path)

# Step 4: Sentiment Analysis (if not already processed)
df = pd.read_csv(cleaned_data_path)
if 'Sentiment Score' not in df.columns:
    print("\nPerforming sentiment analysis on the cleaned dataset...")
    df['Sentiment Score'] = df['Ticket Description'].apply(analyze_sentiment)
    df.to_csv(cleaned_data_path, index=False)
    print("Sentiment scores have been added to the cleaned dataset.")

# Step 5: Wordcloud Generation
print("\nGenerating word cloud...")
text = ' '.join(df['Cleaned Text'].dropna())  # Concatenate all cleaned text
generate_wordcloud(text, wordcloud_output_path)
