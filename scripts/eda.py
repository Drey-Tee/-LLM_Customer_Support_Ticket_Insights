import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(file_path):
    """Load and display basic dataset information."""
    df = pd.read_csv(file_path)
    print("Dataset Overview:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())


def plot_sentiment_distribution(file_path, output_path):
    """Plot distribution of sentiment scores and save as PNG."""
    df = pd.read_csv(file_path)

    plt.figure(figsize=(8, 6))
    sns.histplot(df['Sentiment Score'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()
    print(f"Sentiment distribution plot saved to {output_path}")


def plot_ticket_priority(file_path, output_path):
    """Plot ticket priority counts and save as PNG."""
    df = pd.read_csv(file_path)

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Ticket Priority', palette='viridis')
    plt.title('Ticket Priority Counts')
    plt.xlabel('Priority Level')
    plt.ylabel('Count')
    plt.savefig(output_path)
    plt.close()
    print(f"Ticket priority plot saved to {output_path}")


def plot_ticket_status(file_path, output_path):
    """Plot distribution of ticket status and save as PNG."""
    df = pd.read_csv(file_path)

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Ticket Status', palette='coolwarm')
    plt.title('Distribution of Ticket Status')
    plt.xlabel('Ticket Status')
    plt.ylabel('Count')
    plt.savefig(output_path)
    plt.close()
    print(f"Ticket status distribution plot saved to {output_path}")


def plot_customer_satisfaction(file_path, output_path):
    """Plot distribution of customer satisfaction ratings and save as PNG."""
    df = pd.read_csv(file_path)

    plt.figure(figsize=(8, 6))
    sns.histplot(df['Customer Satisfaction Rating'], bins=10, kde=True, color='green')
    plt.title('Distribution of Customer Satisfaction Ratings')
    plt.xlabel('Customer Satisfaction Rating')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()
    print(f"Customer satisfaction plot saved to {output_path}")




