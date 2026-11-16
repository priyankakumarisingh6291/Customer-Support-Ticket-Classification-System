"""Train and evaluate the ticket classifier (Tasks 3 and 4).

Pipeline:  cleaned text -> TF-IDF (1-2 word n-grams) -> Logistic Regression
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocessing import load_clean, ROOT

MODEL_PATH = ROOT / "model" / "ticket_classifier.joblib"
SEED = 42


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", clf),
    ])


def main():
    df = load_clean(verbose=True)
    X, y = df["clean_description"], df["category"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

    
    print("\n5-fold cross-validation accuracy (training data only):")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(),
    }
    for name, clf in candidates.items():
        s = cross_val_score(make_pipeline(clf), X_train, y_train, cv=cv)
        print(f"  {name:20s} {s.mean():.3f} (+/- {s.std():.3f})")

    # Final model: Logistic Regression (gives probabilities -> confidence score)
    model = make_pipeline(LogisticRegression(max_iter=1000, C=10))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    print(f"\nTEST RESULTS\nAccuracy : {acc:.3f}\nPrecision: {p:.3f} (weighted)\n"
          f"Recall   : {r:.3f} (weighted)\nF1 score : {f:.3f} (weighted)")
    print("\nPer-category report:\n", classification_report(y_test, pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):\n", labels, "\n", cm)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(ROOT / "screenshots" / "confusion_matrix.png", dpi=150); plt.close(fig)

    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    X_test.to_frame().assign(category=y_test).to_csv(ROOT / "model" / "test_set.csv", index=False)
    json.dump({"accuracy": acc, "precision_weighted": p, "recall_weighted": r, "f1_weighted": f,
               "train_size": len(X_train), "test_size": len(X_test), "labels": labels,
               "confusion_matrix": cm.tolist()}, open(ROOT / "model" / "metrics.json", "w"), indent=2)

    
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = vec.get_feature_names_out()
    print("\nTop words per category (largest model weights):")
    for i, c in enumerate(clf.classes_):
        top = clf.coef_[i].argsort()[-6:][::-1]
        print(f"  {c:18s}", ", ".join(names[top]))
    print("\nModel saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
