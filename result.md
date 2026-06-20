# Cristiano Ronaldo Twitter Sentiment Analysis - Research Results

What the World Thinks of Ronaldo: A Twitter Sentiment Analysis and Machine Learning Classification Study.

---

## 1. Executive Summary
This study implements an end-to-end sentiment analysis pipeline to evaluate public opinion towards Cristiano Ronaldo on Twitter (X) during the 2026 FIFA World Cup season. By collecting and analyzing tweets across multiple queries, we clean the text, classify the sentiment (Positive, Negative, Neutral) using VADER, and train machine learning models to predict sentiments. Our research aims to analyze the overall public sentiment distribution and discover the most frequent topics associated with each sentiment category.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Sentiment Distribution
The dataset consists of **1,520** clean, preprocessed tweets. The VADER labeling system categorized the tweets into three sentiment classes using the compound score threshold of `0.05`.

| Sentiment Class | Tweet Count | Percentage |
| :--- | :---: | :---: |
| **Positive** | 645 | 42.43% |
| **Negative** | 467 | 30.72% |
| **Neutral** | 408 | 26.84% |

Public sentiment towards Ronaldo is predominantly positive (42.43%), followed by negative opinion (30.72%), and neutral/factual statements (26.84%).

![Sentiment Distribution](report/fig_distribusi.png)

*Figure 1: Overall Sentiment Distribution of Tweets*

**Insight & Interpretation:**
The distribution reveals that public perception remains largely favorable, with positive sentiment comprising the largest share (42.43%). This highlights Ronaldo's persistent popularity and strong global fanbase during major events. However, the substantial negative sentiment (30.72%) indicates a highly polarized audience. This division is characteristic of high-profile sports icons, where passionate supporters are balanced by active critics discussing his current performance, playstyle, and role in the squad.

---

### 2.2 Sentiment Trend Over Time
The daily sentiment trend chart captures public opinion fluctuations. Outliers (e.g. older tweets from 2025) are dynamically filtered out to focus exclusively on the active June 2026 World Cup match window.

![Sentiment Trend Over Time](report/fig_trend.png)

*Figure 2: Daily Sentiment Trend of Tweets*

**Insight & Interpretation:**
The time-series trend highlights a massive surge in tweet volume on June 17-18, aligning with major matchdays for Portugal. Crucially, positive and negative sentiments spiked simultaneously. This simultaneous spike demonstrates that match events act as major polarization catalysts: a notable play, a missed chance, or a decision to start/substitute Ronaldo triggers intense emotional reactions from both fans (praising his presence) and critics (blaming his performance).

> [!NOTE]
> **Data Spike Explanation:** This dataset was collected using scraping techniques focusing on a specific moment/event in June 2026 (specifically, a major international tournament such as the FIFA World Cup or Euro 2026 occurring during this period). As a result, the vast majority of the data is concentrated within this short window, while the sparse dates from previous years merely represent scattered historical top-ranked tweets retrieved by the search algorithm.

---

### 2.3 Keyword Analysis (WordClouds)
By removing standard English stopwords and proper nouns (`"ronaldo"`, `"cristiano"`, `"cr7"`, `"cr"`), the WordClouds highlight the specific sentiment-bearing words.

#### Positive Sentiment
Focuses on admiration, legacy, and supportive comments.
![Positive Sentiment WordCloud](report/fig_wordcloud_positive.png)

*Figure 3: Most Frequent Words in Positive Tweets*

* **Key Words:** `legend`, `best`, `win`, `dribbler`, `goat`, `score`, `support`, `world cup`.
* **Insight & Interpretation:** The positive WordCloud is dominated by words celebrating Ronaldo's legacy and skills. Terms such as `legend`, `best`, and `goat` reflect his standing as one of the game's greatest figures. The prominence of the word `messi` shows that positive discussions about Ronaldo are frequently framed within the context of his lifelong rivalry with Lionel Messi, as fans compare their career achievements.

#### Negative Sentiment
Captures criticisms, comparisons, and negative reactions.
![Negative Sentiment WordCloud](report/fig_wordcloud_negative.png)

*Figure 4: Most Frequent Words in Negative Tweets*

* **Key Words:** `fail`, `hate`, `selfish`, `old`, `decline`, `wasted`, `blame`, `troll`, `insane`.
* **Insight & Interpretation:** The negative WordCloud reveals criticisms concerning his age and perceived gameplay drawbacks. Key words like `old` and `decline` indicate debates about whether his physical performance is deteriorating. The appearance of `selfish` and `blame` shows that critics often hold him responsible for team setbacks or criticize his positioning and shot selection. Interestingly, `messi` is also prominent here, representing comparisons used by critics to diminish Ronaldo's current impact.

#### Neutral Sentiment
Consists of informational updates, news sharing, and general match stats.
![Neutral Sentiment WordCloud](report/fig_wordcloud_neutral.png)

*Figure 5: Most Frequent Words in Neutral Tweets*

* **Key Words:** `congo`, `portugal`, `game`, `highlight`, `match`, `vs`, `cup`.
* **Insight & Interpretation:** The neutral WordCloud consists of objective, match-related terminology. Words like `game`, `match`, `vs`, and `highlight` point to tweets that share factual updates, scores, and media links. The presence of players like `messi` and `haaland` in this category represents stats-based comparisons and news coverage rather than emotional praise or criticism.

> [!TIP]
> The full frequency analysis of words, active date range details, and other metadata are saved in the text report: [eda_report.txt](report/eda_report.txt).

---

## 3. Modeling and Evaluation

We trained and compared four classification models using TF-IDF features (unigrams, bigrams, and trigrams with `ngram_range=(1,3)`) combined with min-max scaled VADER compound scores calculated on the cleaned text:
1. **Logistic Regression** (Balanced Class Weights)
2. **Multinomial Naive Bayes**
3. **Linear SVM** (Balanced Class Weights)
4. **Random Forest Classifier** (Ensemble)

### 3.1 Model Evaluation Results
Detailed metrics (accuracy, precision, recall, and f1-score) are generated and saved dynamically in `report/classification_report.txt`.

> [!IMPORTANT]
> The detailed accuracy comparison of all candidate models (Logistic Regression, Naive Bayes, Linear SVM, Random Forest) and the classification metrics of the best-selected model can be accessed in: [classification_report.txt](report/classification_report.txt).

* **Confusion Matrix:** The confusion matrix evaluates the predicted labels against the actual VADER ground truth labels.

![Confusion Matrix](report/fig_confusion_matrix.png)

*Figure 6: Confusion Matrix of the Best Classifier Model*

**Insight & Interpretation:**
The confusion matrix for the Logistic Regression model reveals key strengths and challenges of the sentiment classifier:
1. **High Precision for Positive Sentiment:** The model excels at identifying positive sentiment correctly, achieving a high precision of 0.81 (as seen in the classification report). This indicates that the language used by supporters has distinct, easily recognizable patterns.
2. **Neutral/Negative Confusion:** A major source of classification errors lies in distinguishing between neutral and negative/positive sentiments. Because tweets are often concise and context-dependent, statements of fact containing emotionally charged words (or sarcasm) are sometimes misclassified by the linear classifier.
3. **Balanced Performance:** With a balanced recall of 0.71 across all classes, the model avoids favoring the majority class (positive). This shows that the class weight balancing was highly effective in mitigating class imbalance during training.

---

## 4. Key Takeaways & Conclusion
1. **Polarized Legacy:** Although Cristiano Ronaldo maintains a strong base of positive support (42.43%), he faces significant public criticism (30.72% negative) regarding his age (`old`), decline in dribbling (`decline`), and alleged egoistic playstyle (`selfish`, `steal goal`).
2. **Match-Day Impact:** Public volume is highly event-driven, showing huge spikes during Portugal World Cup fixtures, where every action of Ronaldo triggers a wave of polarized social media responses.
3. **Model Success:** By addressing class imbalance, keeping negation terms, removing proper name noise, and extending feature engineering to include VADER compound scores on cleaned text, we successfully built a robust classifier capable of predicting Twitter sentiment.
