# Customer Support Ticket Classification System

**Author:** Priyanka Kumari Singh

## 1. Project overview
A small ML system that reads the description of a customer support ticket and predicts its category
(Login Issue, Application Error, Report, Account Update, Performance, Billing). It also shows a confidence
score and a rule-based suggested reply (bonus task).

## 2. Problem statement
A company receives tickets through many channels. Sorting them by hand is slow, so we learn from historical
tickets and automatically predict the category of a new one.

## 3. Technologies used
Python 3.10+, Pandas, NumPy, Scikit-learn (TF-IDF + Logistic Regression), Matplotlib, Seaborn, Streamlit, joblib,
ReportLab (only to build the PDF report). No external APIs/LLMs are used and there are no keys in the project.

## 4. Dataset
`data/tickets.csv` - **self-created synthetic dataset** (no external source). 198 raw rows, 6 categories,
about 32 tickets per category. Columns: `ticket_id, ticket_description, customer_name, date, category, priority, status`.
Created by `src/make_dataset.py` (descriptions written by hand; priority/status/name/date are random).
Missing values and 6 duplicates were added on purpose to practise data cleaning.
After cleaning: 191 rows.

## 5-6. Installation and dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 7. Train the model (also prints cleaning info, CV comparison, metrics; saves confusion matrix)
```bash
python src/train.py
```
Other scripts: `python src/preprocessing.py` (Task 1 output), `python src/eda.py` (Task 2 charts),
`python src/make_dataset.py` (regenerates the CSV - only if you want to).

## 8. Predict
```bash
python src/predict.py          # interactive
python src/predict.py --test   # runs the 8 documented new-ticket tests
```

## 9. Run the web app
```bash
streamlit run src/app.py
```

## 10. Sample input / output
```
Enter Ticket Description:
> The application is taking too long to load.

Predicted Category: Performance
Confidence: 88.0%
Suggested Response: We are sorry the application feels slow. ...
```

## Results (20% hold-out test set, 39 tickets)
Accuracy 89.7% | Precision 0.917 | Recall 0.897 | F1 0.889 (weighted). 7 of 8 brand-new test tickets were correct.
See `report.pdf` for details, limitations and future improvements.

## Use of AI tools
Claude (Anthropic) was used to help write and structure the code, dataset and documentation.
I reviewed the code and can explain it.
