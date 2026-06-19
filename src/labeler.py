import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Threshold for classifying sentiment based on VADER compound score
# positive: compound >= THRESHOLD
# negative: compound <= -THRESHOLD
# neutral: other
THRESHOLD = 0.05

analyzer = SentimentIntensityAnalyzer()

def label_by_vader(text: str) -> str:
    if not isinstance(text, str):
        return "netral"
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound >= THRESHOLD:
        return "positif"
    elif compound <= -THRESHOLD:
        return "negatif"
    else:
        return "netral"

def main():
    # Prefer translated tweets if available, otherwise fall back to raw
    input_path = "data/raw/tweets_translated.csv"
    if not os.path.exists(input_path):
        input_path = "data/raw/tweets_raw.csv"
        
    output_path = "data/raw/tweets_labeled.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the scraper or translator first.")
        return
        
    print(f"Reading raw tweets from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Labeling tweets using VADER...")
    df["label"] = df["text"].apply(label_by_vader)
    
    # Ensure raw directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully labeled and saved dataset to {output_path}")
    print("\nSentiment Distribution:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
