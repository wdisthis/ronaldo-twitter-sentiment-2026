import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    input_path = "data/processed/tweets_clean.csv"
    output_dir = "report"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the preprocessor first.")
        return
        
    print(f"Reading cleaned dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Drop rows with missing values in target columns
    df.dropna(subset=["text_clean", "label"], inplace=True)
    
    if len(df) < 10:
        print("Error: Dataset is too small to split and train models. Please scrape more data.")
        return
        
    # Check class distribution to determine if stratification is possible
    class_counts = df["label"].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # Check if any class has only 1 sample; if so, we can't stratify
    can_stratify = all(count >= 2 for count in class_counts)
    stratify_param = df["label"] if can_stratify else None
    if not can_stratify:
        print("Warning: Some classes have fewer than 2 samples. Disabling stratification.")
        
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify_param
    )
    
    y_train = df_train["label"]
    y_test  = df_test["label"]
    
    print(f"\nTraining set size: {len(df_train)}")
    print(f"Testing set size: {len(df_test)}")
    
    # Vectorization using TF-IDF (unigrams and bigrams)
    print("\nExtracting features using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_train_tfidf = tfidf.fit_transform(df_train["text_clean"])
    X_test_tfidf  = tfidf.transform(df_test["text_clean"])
    
    # Feature Engineering: VADER Polarity Scores (calculated on raw text)
    print("Extracting VADER features...")
    analyzer = SentimentIntensityAnalyzer()
    
    def get_vader_features(dataframe):
        feats = []
        for text in dataframe["text_clean"]:
            if not isinstance(text, str):
                feats.append([0.5])  # neutral
            else:
                compound = analyzer.polarity_scores(text)["compound"]
                # Scale from [-1, 1] to [0, 1] for MultinomialNB compatibility
                scaled = (compound + 1.0) / 2.0
                feats.append([scaled])
        return csr_matrix(feats)
        
    X_train_extra = get_vader_features(df_train)
    X_test_extra  = get_vader_features(df_test)
    
    # Combine TF-IDF and VADER features
    X_train_v = hstack([X_train_tfidf, X_train_extra])
    X_test_v  = hstack([X_test_tfidf, X_test_extra])
    
    # Model Candidates
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Naive Bayes":         MultinomialNB(),
        "Linear SVM":          LinearSVC(class_weight='balanced'),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    # Train and evaluate candidates
    best_name, best_model, best_score = None, None, 0
    print("\nComparing models:")
    for name, clf in candidates.items():
        try:
            clf.fit(X_train_v, y_train)
            score = clf.score(X_test_v, y_test)
            print(f" - {name}: Accuracy = {score:.4f}")
            if score > best_score:
                best_name, best_model, best_score = name, clf, score
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    if best_model is None:
        print("Error: No models were trained successfully.")
        return
        
    print(f"\nBest Model selected: {best_name} (Accuracy: {best_score:.4f})")
    
    # Detailed Evaluation
    y_pred = best_model.predict(X_test_v)
    clf_rep = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(clf_rep)
    
    # Ensure report directory exists for saving evaluation plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the classification report to a text file in laporan/
    report_path = os.path.join(output_dir, "classification_report.txt")
    try:
        # Build model comparison summary string
        comp_summary = ""
        for name, clf in candidates.items():
            if clf == best_model:
                comp_summary += f" - {name:<20}: Accuracy = {clf.score(X_test_v, y_test):.4f} (Selected Best)\n"
            else:
                comp_summary += f" - {name:<20}: Accuracy = {clf.score(X_test_v, y_test):.4f}\n"
                
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write("         CLASSIFICATION REPORT           \n")
            f.write("=========================================\n\n")
            f.write("1. Model Performance Comparison\n")
            f.write("-------------------------------\n")
            f.write(comp_summary)
            f.write("\n")
            f.write("2. Final Classification Report (Best Model)\n")
            f.write("------------------------------------------\n")
            f.write(f"Model Name: {best_name}\n")
            f.write(f"Accuracy  : {best_score:.4f}\n\n")
            f.write(clf_rep)
            f.write("\n=========================================\n")
        print(f"Classification report successfully saved to '{report_path}'.")
    except Exception as e:
        print(f"Could not save classification report: {e}")
    
    # Confusion Matrix
    print("Generating Confusion Matrix Heatmap...")
    labels = sorted(list(df["label"].unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Save the model and vectorizer in models/ directory for clean structure
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")
    tfidf_path = os.path.join(models_dir, "tfidf.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
        
    print(f"\nModel artifacts successfully saved to '{model_path}' and '{tfidf_path}'")

if __name__ == "__main__":
    main()
