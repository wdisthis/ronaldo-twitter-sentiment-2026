import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def main():
    input_path = "data/processed/tweets_clean.csv"
    output_dir = "laporan"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the preprocessor first.")
        return
        
    print(f"Reading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure laporan directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sentiment Distribution Chart
    print("Generating Sentiment Distribution Chart...")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts()
    
    # Define custom colors corresponding to [Negative, Neutral, Positive]
    color_map = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}
    bar_colors = [color_map.get(x, "#3498db") for x in counts.index]
    
    counts.plot(kind="bar", color=bar_colors, ax=ax)
    ax.set_title("Sentiment Distribution of Tweets about Ronaldo")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Tweet Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_distribusi.png"), dpi=150)
    plt.close()
    
    # 2. WordCloud for each sentiment
    print("Generating WordClouds for each sentiment class...")
    for label in ["positive", "negative", "neutral"]:
        subset = df[df["label"] == label]["text_clean"]
        # Drop NaN or empty texts
        subset = subset.dropna()
        corpus = " ".join(subset.astype(str))
        
        if corpus.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(corpus)
            plt.figure(figsize=(8, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"WordCloud - {label.capitalize()} Sentiment")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"fig_wordcloud_{label}.png"), dpi=150)
            plt.close()
        else:
            print(f"Skipping WordCloud for '{label}' - corpus is empty.")
            
    # 3. Sentiment Trend Over Time
    if "created_at" in df.columns:
        print("Generating Sentiment Trend Over Time Chart...")
        try:
            # Filter out any duplicate header rows if present
            df_filtered = df[df["created_at"] != "created_at"].copy()
            # Convert and coerce errors to NaT
            df_filtered["date"] = pd.to_datetime(df_filtered["created_at"], errors="coerce").dt.date
            df_trend = df_filtered.dropna(subset=["date"])
            
            # Filter out date outliers dynamically to focus on the active time range
            if not df_trend.empty:
                dates_dt = pd.to_datetime(df_trend["date"])
                q_start = dates_dt.quantile(0.05).date()
                q_end = dates_dt.quantile(0.95).date()
                start_date = q_start - pd.Timedelta(days=1)
                end_date = q_end + pd.Timedelta(days=1)
                df_trend = df_trend[(df_trend["date"] >= start_date) & (df_trend["date"] <= end_date)]
            
            trend = df_trend.groupby(["date", "label"]).size().unstack(fill_value=0)
            # Format date index as 'DD Month' (e.g. 18 Jun) since all data is from 2026
            trend.index = pd.to_datetime(trend.index).strftime("%d %b")
            
            # Reorder columns to match standard colors if present
            ordered_cols = [col for col in ["negative", "neutral", "positive"] if col in trend.columns]
            trend_colors = [color_map.get(col) for col in ordered_cols]
            
            trend[ordered_cols].plot(figsize=(10, 5), color=trend_colors)
            plt.title("Sentiment Trend of Tweets about Ronaldo per Day")
            plt.xlabel("Date")
            plt.ylabel("Tweet Count")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig_trend.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not generate trend chart: {e}")
            
    print(f"All visualizations successfully saved in '{output_dir}/' directory.")

if __name__ == "__main__":
    main()
