"""Predict the category of a new ticket (Task 5, command-line interface = Task 6 Option A).

Usage:
    python src/predict.py            # interactive prompt
    python src/predict.py --test     # run the 8 documented new-ticket test cases
"""
import sys
import joblib
from preprocessing import clean_text, ROOT
from responses import suggest_response

MODEL_PATH = ROOT / "model" / "ticket_classifier.joblib"


TEST_CASES = [
    ("I am unable to login because my password is not working.", "Login Issue"),
    ("The app shows an error whenever I press save", "Application Error"),
    ("Need a sales report for last week", "Report"),
    ("Kindly change my phone number registered with the account", "Account Update"),
    ("Everything is running so slowly right now", "Performance"),
    ("I was billed two times for my subscription this month", "Billing"),
    ("My account is locked and I cannot sign in", "Login Issue"),
    ("Please send the revenue summary for the quarter", "Report"),
]


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run:  python src/train.py  first.")
    return joblib.load(MODEL_PATH)


def predict(model, text: str):
    """Return (category, confidence %, all probabilities)."""
    probs = model.predict_proba([clean_text(text)])[0]
    i = probs.argmax()
    return model.classes_[i], probs[i] * 100, dict(zip(model.classes_, probs))


def run_tests(model):
    ok = 0
    print(f"{'Test Input':58s} {'Expected':18s} {'Predicted':18s} Conf.   Result")
    print("-" * 112)
    for text, expected in TEST_CASES:
        cat, conf, _ = predict(model, text)
        ok += cat == expected
        print(f"{text[:57]:58s} {expected:18s} {cat:18s} {conf:5.1f}%  {'PASS' if cat == expected else 'FAIL'}")
    print(f"\n{ok}/{len(TEST_CASES)} correct")


def main():
    model = load_model()
    if "--test" in sys.argv:
        return run_tests(model)
    print("=" * 36, "\n Customer Support Classifier\n", "=" * 36, sep="")
    while True:
        text = input("\nEnter Ticket Description (or 'q' to quit):\n> ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            print("Please type something."); continue
        cat, conf, _ = predict(model, text)
        print(f"\nPredicted Category: {cat}\nConfidence: {conf:.1f}%\nSuggested Response: {suggest_response(cat)}")


if __name__ == "__main__":
    main()
