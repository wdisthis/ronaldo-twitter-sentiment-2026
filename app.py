# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set page configuration first
st.set_page_config(
    page_title="Ronaldo Twitter Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# --- Theme & Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');
    
    /* Global Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metrics Custom CSS */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-color);
    }
    .metric-title {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.8;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        color: var(--text-color);
    }
    .metric-subtitle {
        font-size: 0.8rem;
        margin-top: 8px;
        opacity: 0.7;
    }
    
    /* Prediction output card */
    .pred-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        border: 2px solid;
    }
    .pred-positive {
        background-color: rgba(46, 204, 113, 0.1);
        border-color: #2ecc71;
        color: #2ecc71;
    }
    .pred-negative {
        background-color: rgba(231, 76, 60, 0.1);
        border-color: #e74c3c;
        color: #e74c3c;
    }
    .pred-neutral {
        background-color: rgba(149, 165, 166, 0.1);
        border-color: #95a5a6;
        color: #95a5a6;
    }
    
    /* Footer design */
    .footer {
        text-align: center;
        padding: 30px 0;
        margin-top: 50px;
        font-size: 0.85rem;
        opacity: 0.7;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Asset Caching ---
@st.cache_resource
def load_model():
    model_path = "models/model.pkl"
    tfidf_path = "models/tfidf.pkl"
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
        return model, tfidf, True
    return None, None, False

@st.cache_data
def load_data():
    csv_path = "data/processed/tweets_clean.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Drop rows that are duplicate headers
        if "likes" in df.columns:
            df = df[df["likes"] != "likes"].copy()
            df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
        if "retweets" in df.columns:
            df["retweets"] = pd.to_numeric(df["retweets"], errors="coerce").fillna(0).astype(int)
        return df, True
    return None, False

# Load assets
model, tfidf, model_loaded = load_model()
df_raw, data_loaded = load_data()

# Text Preprocessing settings (keep consistent with preprocessor.py)
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "but", "against", "without",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", 
    "hadnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt"
}
CUSTOM_STOPWORDS = {"ronaldo", "cristiano", "cr7", "cr"}
stop_words = (set(stopwords.words("english")) - NEGATION_WORDS) | CUSTOM_STOPWORDS
stemmer = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # Remove numbers and symbols
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# --- Header Section ---
col_logo, col_title = st.columns([1, 12])
with col_title:
    st.title("Cristiano Ronaldo Twitter Sentiment Analysis")
    st.markdown("Exploring the public narrative, sentiment trends, and machine learning predictions of tweets surrounding CR7 during the 2026 World Cup / Euro season.")

# Check if data exists; if not, show setup warning
if not data_loaded:
    st.error("⚠️ Labeled and preprocessed dataset not found!")
    st.markdown("""
    Please run the preprocessing pipeline first:
    ```bash
    python src/scraper.py
    python src/translator.py
    python src/labeler.py
    python src/preprocessor.py
    python src/report.py
    python src/train.py
    ```
    """)
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")
st.sidebar.markdown("Filter the dashboard dataset interactively:")

# 1. Query selection
all_queries = sorted(df_raw["query"].dropna().unique())
query_filter = st.sidebar.multiselect(
    "Query Source:",
    all_queries,
    default=all_queries
)

# 2. Date range selection
df_raw["created_at_parsed"] = pd.to_datetime(df_raw["created_at"], errors="coerce")
df_time_clean = df_raw.dropna(subset=["created_at_parsed"])

if not df_time_clean.empty:
    min_date = df_time_clean["created_at_parsed"].min().date()
    max_date = df_time_clean["created_at_parsed"].max().date()
    
    date_filter = st.sidebar.date_input(
        "Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_filter = None

# 3. Likes/Retweets range
max_likes = int(df_raw["likes"].max()) if "likes" in df_raw.columns else 1000
likes_filter = st.sidebar.slider(
    "Minimum Likes:",
    min_value=0,
    max_value=1000 if max_likes < 1000 else max_likes,
    value=0
)

# Apply filters
df_filtered = df_raw.copy()
if query_filter:
    df_filtered = df_filtered[df_filtered["query"].isin(query_filter)]

if date_filter and len(date_filter) == 2:
    start_d, end_d = date_filter
    df_filtered["date_only"] = df_filtered["created_at_parsed"].dt.date
    df_filtered = df_filtered[(df_filtered["date_only"] >= start_d) & (df_filtered["date_only"] <= end_d)]

if "likes" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["likes"] >= likes_filter]

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & Analytics", 
    "WordCloud & Text Explorer", 
    "Real-time Prediction", 
    "Model performance"
])

# --- TAB 1: Overview & Analytics ---
with tab1:
    st.subheader("Key Metrics Overview")
    
    total_tweets = len(df_filtered)
    counts = df_filtered["label"].value_counts()
    pos_count = counts.get("positive", 0)
    neg_count = counts.get("negative", 0)
    neu_count = counts.get("neutral", 0)
    
    pos_pct = (pos_count / total_tweets * 100) if total_tweets > 0 else 0
    neg_pct = (neg_count / total_tweets * 100) if total_tweets > 0 else 0
    neu_pct = (neu_count / total_tweets * 100) if total_tweets > 0 else 0
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Filtered Tweets</div>
            <div class="metric-value">{total_tweets:,}</div>
            <div class="metric-subtitle">from active filters</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Positive Sentiment</div>
            <div class="metric-value">{pos_pct:.1f}%</div>
            <div class="metric-subtitle">{pos_count:,} tweets</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Neutral Sentiment</div>
            <div class="metric-value">{neu_pct:.1f}%</div>
            <div class="metric-subtitle">{neu_count:,} tweets</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Negative Sentiment</div>
            <div class="metric-value">{neg_pct:.1f}%</div>
            <div class="metric-subtitle">{neg_count:,} tweets</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    c_col1, c_col2 = st.columns([1, 1])
    
    with c_col1:
        st.subheader("Sentiment Distribution")
        if total_tweets > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            # Match colors with report.py
            color_map = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
            
            # Reorder labels to match standard order
            labels_order = [lbl for lbl in ["positive", "neutral", "negative"] if lbl in counts.index]
            counts_ordered = counts[labels_order]
            colors = [color_map.get(lbl, "#3498db") for lbl in labels_order]
            
            # Style matplotlib beautifully
            fig.patch.set_facecolor('#ffffff')
            ax.patch.set_facecolor('#ffffff')
            sns.barplot(x=counts_ordered.index, y=counts_ordered.values, palette=colors, ax=ax, hue=counts_ordered.index, legend=False)
            
            ax.set_title("Tweet Counts by Sentiment", fontsize=11, fontweight="bold", color="#2c3e50")
            ax.set_ylabel("Tweet Count", fontsize=9, color="#2c3e50")
            ax.tick_params(colors="#2c3e50")
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No data available for the current filter.")
            
    with c_col2:
        st.subheader("Sentiment Volume Trend over Time")
        # Preprocess time trend data
        df_trend = df_filtered.dropna(subset=["created_at_parsed"]).copy()
        if not df_trend.empty:
            # Group by day & label
            df_trend["date"] = df_trend["created_at_parsed"].dt.date
            
            # Dynamic date range filter for trend line chart (trim extremes to focus on core tournament)
            dates_dt = pd.to_datetime(df_trend["date"])
            q_start = dates_dt.quantile(0.05).date()
            q_end = dates_dt.quantile(0.95).date()
            start_date = q_start - pd.Timedelta(days=1)
            end_date = q_end + pd.Timedelta(days=1)
            df_trend = df_trend[(df_trend["date"] >= start_date) & (df_trend["date"] <= end_date)]
            
            trend_df = df_trend.groupby(["date", "label"]).size().unstack(fill_value=0)
            
            # Reorder trend columns
            ordered_cols = [col for col in ["negative", "neutral", "positive"] if col in trend_df.columns]
            trend_df = trend_df[ordered_cols]
            
            # Rename columns for legend
            column_mapping = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
            trend_df.rename(columns=column_mapping, inplace=True)
            
            st.line_chart(trend_df, use_container_width=True)
        else:
            st.info("No trend data available for current date filters.")

    st.subheader("Labeled Corpus Table")
    st.markdown("Filter and view raw tweets collected in the dataset:")
    st.dataframe(
        df_filtered[["username", "created_at", "likes", "retweets", "query", "label", "text"]],
        use_container_width=True,
        hide_index=True
    )

# --- TAB 2: WordCloud & Text Explorer ---
with tab2:
    st.subheader("WordCloud Insights")
    st.markdown("Visualizing key words associated with different sentiments. Note: common stop words and player names (*Ronaldo, Cristiano, CR7*) are filtered to highlight sentiment-bearing words.")
    
    wc_label = st.radio("Choose Sentiment Class:", ["positive", "neutral", "negative"], horizontal=True)
    
    # Filter corpus based on sentiment
    df_wc = df_filtered[df_filtered["label"] == wc_label]
    corpus = " ".join(df_wc["text_clean"].dropna())
    
    if corpus.strip():
        # Setup WordCloud plot
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        fig_wc.patch.set_alpha(0.0)
        ax_wc.patch.set_alpha(0.0)
        
        # Color palettes for WordClouds
        colormap_map = {
            "positive": "Greens",
            "neutral": "Blues",
            "negative": "Reds"
        }
        
        wc = WordCloud(
            width=800, 
            height=300, 
            background_color=None, 
            mode="RGBA",
            colormap=colormap_map.get(wc_label, "viridis")
        ).generate(corpus)
        
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        plt.tight_layout()
        st.pyplot(fig_wc)
    else:
        st.info(f"No word cloud data available for sentiment: {wc_label}")
        
    st.markdown("---")
    
    st.subheader(f"Top {wc_label.capitalize()} Tweets")
    st.markdown("High-engagement tweets classified under this category:")
    df_wc_top = df_wc.sort_values(by="likes", ascending=False).head(10)
    
    for idx, row in df_wc_top.iterrows():
        emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😡"}
        likes_count = int(row['likes']) if not pd.isna(row['likes']) else 0
        rts_count = int(row['retweets']) if not pd.isna(row['retweets']) else 0
        
        st.markdown(f"""
        <div style="background-color: var(--secondary-background-color); border-left: 5px solid {color_map[wc_label]}; padding: 15px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(128, 128, 128, 0.1);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; opacity: 0.8;">
                <strong>@{row['username']}</strong>
                <span>Likes: {likes_count:,} | RTs: {rts_count:,}</span>
            </div>
            <div style="font-size: 0.95rem; line-height: 1.4; color: var(--text-color);">"{row['text']}"</div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: Real-time Prediction ---
with tab3:
    st.subheader("Predict Custom Tweet Sentiment")
    st.markdown("Enter any custom tweet or text snippet to analyze its sentiment using the trained Logistic Regression model stacked with VADER compound scores.")
    
    # Warning if model is not trained yet
    if not model_loaded:
        st.warning("Machine learning model artifacts not found! Real-time prediction is disabled.")
        st.info("Run `python src/train.py` to train models and save PKL files.")
    else:
        input_text = st.text_area(
            "Tweet Text:", 
            placeholder="Type something about Cristiano Ronaldo (e.g. 'Ronaldo is the greatest of all time, unmatched legend!')",
            height=120
        )
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            predict_clicked = st.button("Analyze Sentiment", use_container_width=True)
            
        if predict_clicked:
            if input_text.strip():
                with st.spinner("Analyzing tweet features and running classifier..."):
                    # 1. Clean the text
                    clean_text = preprocess(input_text)
                    
                    # 2. Extract TF-IDF features
                    tfidf_feat = tfidf.transform([clean_text])
                    
                    # 3. Calculate VADER polarity
                    vader_scores = analyzer.polarity_scores(input_text)
                    compound = vader_scores["compound"]
                    scaled_vader = (compound + 1.0) / 2.0
                    vader_feat = csr_matrix([[scaled_vader]])
                    
                    # 4. Stack features
                    vec = hstack([tfidf_feat, vader_feat])
                    
                    # 5. Predict
                    prediction = model.predict(vec)[0]
                    probs = model.predict_proba(vec)[0]
                    classes = model.classes_
                    prob_dict = dict(zip(classes, probs))
                    
                    # Sentiment formatting
                    emoji_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
                    class_styles = {
                        "positive": "pred-positive",
                        "neutral": "pred-neutral",
                        "negative": "pred-negative"
                    }
                    
                    # Show prediction output card
                    st.markdown(f"""
                    <div class="pred-card {class_styles[prediction]}">
                        <div style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.8; margin-bottom: 5px;">Prediction Results</div>
                        <div style="font-size: 2.2rem; font-weight: 800;">{emoji_map[prediction].upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Classification Probability Breakdown")
                    # Visual representation of probabilities
                    pb_col1, pb_col2, pb_col3 = st.columns(3)
                    with pb_col1:
                        st.metric("Positive Probability", f"{prob_dict.get('positive', 0)*100:.1f}%")
                        st.progress(float(prob_dict.get('positive', 0)))
                    with pb_col2:
                        st.metric("Neutral Probability", f"{prob_dict.get('neutral', 0)*100:.1f}%")
                        st.progress(float(prob_dict.get('neutral', 0)))
                    with pb_col3:
                        st.metric("Negative Probability", f"{prob_dict.get('negative', 0)*100:.1f}%")
                        st.progress(float(prob_dict.get('negative', 0)))
                        
                    st.markdown("### Pipeline Extraction Details")
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        st.markdown("**Clean Text (Input to TF-IDF)**")
                        st.code(clean_text if clean_text else "[Empty after filtering stopwords]")
                    with p_col2:
                        st.markdown("**VADER Polarity Scores**")
                        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                        v_col1.metric("Pos Score", f"{vader_scores['pos']:.3f}")
                        v_col2.metric("Neu Score", f"{vader_scores['neu']:.3f}")
                        v_col3.metric("Neg Score", f"{vader_scores['neg']:.3f}")
                        v_col4.metric("Compound (Scaled)", f"{compound:.3f} ({scaled_vader:.3f})")
            else:
                st.warning("Please type a valid tweet to analyze.")

# --- TAB 4: Model Performance ---
with tab4:
    st.subheader("Model Training & Performance Comparison")
    st.markdown("This tab displays evaluation reports and metrics generated from training the machine learning pipeline on stacked TF-IDF and VADER features.")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.subheader("Classifier Classification Report")
        report_txt_path = "report/classification_report.txt"
        if os.path.exists(report_txt_path):
            with open(report_txt_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.text_area("classification_report.txt", report_content, height=350, disabled=True)
        else:
            st.info("Classification report file not found in `report/` directory.")
            
    with perf_col2:
        st.subheader("Confusion Matrix")
        matrix_img_path = "report/fig_confusion_matrix.png"
        if os.path.exists(matrix_img_path):
            st.image(matrix_img_path, caption="Confusion Matrix for the Selected Best Classifier (Logistic Regression)")
        else:
            st.info("Confusion matrix plot not found in `report/` directory.")

# Footer section
st.markdown("""
<div class="footer">
    <p>Cristiano Ronaldo Twitter Sentiment Analysis Dashboard © 2026. Made with Streamlit, Python & Scikit-learn.</p>
</div>
""", unsafe_allow_html=True)
