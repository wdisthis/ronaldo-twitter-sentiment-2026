import os
import time
import pandas as pd
from deep_translator import GoogleTranslator

def main():
    input_path = "data/raw/tweets_raw.csv"
    output_path = "data/raw/tweets_translated.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the scraper first.")
        return
        
    print(f"Reading raw tweets from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter out empty texts
    df = df[df["text"].notna()]
    
    # We only need to translate non-English tweets
    non_en_mask = (df["lang"] != "en")
    total_to_translate = non_en_mask.sum()
    print(f"Found {total_to_translate} non-English tweets to translate.")
    
    if total_to_translate == 0:
        print("All tweets are already in English.")
        df.to_csv(output_path, index=False)
        return
        
    translator = GoogleTranslator(source="auto", target="en")
    
    print("Translating non-English tweets to English...")
    translated_count = 0
    
    # Backup original text to a new column
    if "text_original" not in df.columns:
        df["text_original"] = df["text"]
        
    for idx, row in df[non_en_mask].iterrows():
        original_text = row["text"]
        try:
            # Perform translation
            translated_text = translator.translate(original_text)
            df.at[idx, "text"] = translated_text
            # Mark lang as "en" so VADER and preprocessor can process it as English
            df.at[idx, "lang"] = "en"
            translated_count += 1
            
            # Print progress
            if translated_count % 10 == 0 or translated_count == total_to_translate:
                print(f"Progress: Translated {translated_count}/{total_to_translate} tweets...")
                
            # Sleep briefly to respect translation API rate limits
            time.sleep(0.2)
        except Exception as e:
            print(f"Error translating tweet ID {row['tweet_id']}: {e}")
            time.sleep(1.0)
            
    df.to_csv(output_path, index=False)
    print(f"\nTranslation complete! Saved {translated_count} translated tweets to '{output_path}'.")

if __name__ == "__main__":
    main()
