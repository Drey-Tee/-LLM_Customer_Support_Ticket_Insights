from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(text, output_path):
    """Generate a word cloud from the text and save it as an image."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(output_path)
    print(f"Word cloud saved to {output_path}")
