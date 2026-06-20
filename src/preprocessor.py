import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available locally
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords")

stemmer    = PorterStemmer()
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "but", "against", "without",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", 
    "hadnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt"
}
CUSTOM_STOPWORDS = {"ronaldo", "cristiano", "messi", "cr7", "cr"}
stop_words = (set(stopwords.words("english")) - NEGATION_WORDS) | CUSTOM_STOPWORDS

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # Remove numbers and symbols
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    # Filter stopwords and short terms
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Apply stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def main():
    input_path = "data/raw/tweets_labeled.csv"
    output_path = "data/processed/tweets_clean.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the labeler first.")
        return
        
    print(f"Reading labeled tweets from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter for English and Indonesian languages
    print("Filtering languages...")
    df = df[df["lang"].isin(["en", "in"])]
    df.dropna(subset=["text", "label"], inplace=True)
    
    print("Cleaning and preprocessing tweet text...")
    df["text_clean"] = df["text"].apply(preprocess)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Cleaned dataset saved to {output_path} with {len(df)} records.")

if __name__ == "__main__":
    main()
