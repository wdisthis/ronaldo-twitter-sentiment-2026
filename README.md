# Ronaldo Twitter Sentiment Analysis 2026

What the World Thinks of Ronaldo in 2026: A Twitter Sentiment Analysis During the FIFA World Cup Season.

This project implements an end-to-end sentiment analysis pipeline: Scraping Twitter -> Preprocessing -> Modeling -> Reports and Dashboard.

## Project Goals

The primary goal of this project is to answer the question:
> "How is the public sentiment on Twitter towards Ronaldo, and is the majority negative compared to other players like Messi, Haaland, and Mbappe?"

### Expected Outputs

| Output | Format | Description |
|---|---|---|
| Clean Dataset | `data/processed/tweets_clean.csv` | Labeled sentiment tweets, ready for analysis |
| Trained Model | `model.pkl` + `tfidf.pkl` | 3-class sentiment classifier |
| Analysis Report | Plots in `laporan/` | Distribution charts, wordclouds, and evaluation metrics |
| Interactive Dashboard | Streamlit app (`app.py`) | Visualization of sentiment trends, keyword filtering, and date filtering |

## Project Structure

```text
ronaldo-sentiment/
├── data/
│   ├── raw/                  # Raw tweets scraped from Twitter (.csv)
│   └── processed/            # Processed and labeled tweets
├── laporan/                  # Generated plots and reports
├── src/
│   ├── scraper.py            # Twitter scraper using twikit
│   ├── preprocessor.py       # Text cleaning and stemming
│   ├── labeler.py            # Sentiment labeling (lexicon or manual)
│   ├── train.py              # Model training and evaluation
│   └── report.py             # Script to generate report plots
├── app.py                    # Streamlit dashboard application
├── requirements.txt
└── README.md
```

## Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd ronaldo-twitter-sentiment-2026
   ```

2. **Create and Activate a Virtual Environment**
   It is highly recommended to use a virtual environment to isolate project dependencies:

   * **Windows (PowerShell):**
     ```powershell
     python -m venv .env
     .\.env\Scripts\Activate.ps1
     ```

   * **Windows (Command Prompt):**
     ```cmd
     python -m venv .env
     .env\Scripts\activate.bat
     ```

   * **macOS / Linux:**
     ```bash
     python3 -m venv .env
     source .env/bin/activate
     ```

3. **Install Dependencies**
   Once the virtual environment is active, run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Twikit Credentials**
   Open `src/scraper.py` and configure your Twitter credentials:
   ```python
   USERNAME = "your_twitter_username"
   EMAIL    = "your_email@gmail.com"
   PASSWORD = "your_password"
   ```

## Pipeline Execution

Run the scripts in the following order to execute the full pipeline:

### 1. Data Scraping
Scrapes tweets from Twitter using Twikit and saves them to `data/raw/tweets_raw.csv`.
```bash
python src/scraper.py
```

### 2. Sentiment Labeling
Labels raw tweets into positive, negative, or neutral classes using a lexicon-based approach.
```bash
python src/labeler.py
```

### 3. Text Preprocessing
Cleans raw text (removes URLs, hashtags, mentions, digits, and stopwords), applies stemming, and saves clean data to `data/processed/tweets_clean.csv`.
```bash
python src/preprocessor.py
```

### 4. Exploratory Data Analysis & Report Generation
Generates distribution charts, wordclouds, and daily sentiment trend lines, saving them to the `laporan/` folder.
```bash
python src/report.py
```

### 5. Model Training
Trains three classification models (Logistic Regression, Naive Bayes, Linear SVM) using TF-IDF features, compares performance, and saves `model.pkl` and `tfidf.pkl`.
```bash
python src/train.py
```

### 6. Run the Interactive Dashboard
Launch the Streamlit dashboard to explore sentiment trends and test sentiment prediction on custom tweets.
```bash
streamlit run app.py
```

## Project Progress Checklist

- [x] Set up environment and install required libraries
- [x] Log in via Twikit and save cookies.json
- [ ] Scrape tweets (target: 2000+ unique tweets)
- [ ] Perform sentiment labeling (lexicon or VADER)
- [ ] Preprocess texts and save to tweets_clean.csv
- [ ] Perform EDA and save generated graphics to laporan/
- [ ] Train classifier models and save model.pkl and tfidf.pkl
- [ ] Evaluate the best model (classification report and confusion matrix)
- [ ] Run and test the Streamlit dashboard locally
