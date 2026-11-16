"""Builds report.pdf from the saved metrics and screenshots (Task 7)."""
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from preprocessing import ROOT
from predict import TEST_CASES, load_model, predict

m = json.load(open(ROOT / "model" / "metrics.json"))
S = getSampleStyleSheet()
B = ParagraphStyle("b", parent=S["BodyText"], fontSize=10, leading=14)
H = S["Heading2"]


def P(t): return Paragraph(t, B)
def bullets(items): return [Paragraph("&bull; " + i, B) for i in items]
def tbl(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t
def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(ROOT / "screenshots" / name).size
    return Image(str(ROOT / "screenshots" / name), width=w, height=w * ih / iw)

model = load_model()
rows = [["Test Input", "Expected", "Predicted", "Conf.", "Result"]]
for t, e in TEST_CASES:
    c, conf, _ = predict(model, t)
    rows.append([Paragraph(t, B), e, c, f"{conf:.1f}%", "Correct" if c == e else "Wrong"])

acc = m["accuracy"] * 100
s = [Paragraph("Customer Support Ticket Classification - Solution Report", S["Title"]),
     P("Priyanka Kumari Singh - AI Fresher Technical Assignment"), Spacer(1, 10)]
s += [Paragraph("A. Problem Understanding", H),
      P("A company gets many support tickets. Reading and sorting each one by hand is slow. The goal is a system that "
        "learns from old tickets and predicts the category of a new ticket from its description alone, so tickets can be "
        "routed to the right team faster.")]
s += [Paragraph("B. Dataset", H),
      P("No public dataset was used. I created a <b>synthetic dataset</b> (src/make_dataset.py): 198 raw records, "
        "6 categories (Login Issue, Application Error, Report, Account Update, Performance, Billing), about 32 hand-written "
        "descriptions per category. Fields: ticket_id, ticket_description, customer_name, date, category, priority, status. "
        "Priority, status, names and dates are random (priority is skewed by category), so they carry little real meaning. "
        "A few missing values and duplicates were inserted deliberately to practise cleaning.")]
s += [Paragraph("C. Data Preprocessing", H)] + bullets([
    "<b>Missing values:</b> 1 ticket had no description - dropped (cannot learn from it). Missing priority/status/name "
    "(10 cells) were filled with 'Unknown' because they are not used by the model.",
    "<b>Duplicates:</b> 6 duplicates (some differing only in CAPITALS or spaces) were removed after cleaning the text. 198 -> 191 rows.",
    "<b>Text cleaning:</b> lowercase, remove punctuation/digits, collapse extra spaces. Stop-words were kept because words "
    "like 'not' and 'cannot' matter (e.g. 'password reset is not working').",
    "<b>Why:</b> the computer treats 'Password' and 'password.' as different words; cleaning makes equal words match. "
    "Removing duplicates stops the same ticket appearing in both train and test sets, which would inflate accuracy."])
s += [Paragraph("Exploratory Data Analysis", H),
      P("191 tickets in total. Categories are almost perfectly balanced (31-32 each, ~16.7%), so accuracy is a fair metric. "
        "Priority: Medium 87, High 62, Low 38, Unknown 4. Status: Open 78, Closed 65, In Progress 46, Unknown 2. "
        "Descriptions are short (3-13 words, mean 8.9)."),
      img("category_distribution.png", 11 * cm), img("priority_distribution.png", 11 * cm),
      P("<b>Observation:</b> Login Issue, Application Error and Performance tickets are mostly High priority, while Report "
        "and Account Update tickets are mostly Medium/Low. This is because I generated priorities that way, so in a real "
        "company this should be re-checked on real data.")]
s += [PageBreak(), Paragraph("D. Model Selection", H),
      P("<b>Pipeline:</b> Ticket text -> cleaning -> TF-IDF (single words and 2-word pairs) -> Logistic Regression -> category."),
      P("<b>Features:</b> TF-IDF gives each word a number: higher if it appears often in this ticket but is rare in other "
        "tickets (e.g. 'password', 'invoice'), lower for common words ('the'). Each ticket becomes a vector of such numbers."),
      P("<b>Algorithm:</b> Logistic Regression. It learns a weight for every word per category; for a new ticket it adds up the "
        "weights and converts the scores to probabilities (softmax). The category with the highest probability is the prediction, "
        "and that probability is the confidence. I chose it because it is simple, fast on small data, works well with TF-IDF, "
        "gives probabilities, and is explainable (I can print the top words per category). In 5-fold cross-validation on the "
        "training data: Logistic Regression 0.855, Linear SVM 0.848, Naive Bayes 0.796.")]
s += [Paragraph("E. Training and Testing", H),
      P(f"Data was split 80/20 with a fixed seed and stratification (same category mix in both): {m['train_size']} training and "
        f"{m['test_size']} test tickets. The model was fitted only on the training set. The test set was used once, at the end, "
        "to measure performance on tickets the model had never seen.")]
s += [Paragraph("F. Results", H),
      tbl([["Metric", "Value"], ["Accuracy", f"{acc:.1f}%"], ["Precision (weighted)", f"{m['precision_weighted']:.3f}"],
           ["Recall (weighted)", f"{m['recall_weighted']:.3f}"], ["F1 score (weighted)", f"{m['f1_weighted']:.3f}"]],
          [6 * cm, 4 * cm]), Spacer(1, 6), img("confusion_matrix.png", 10 * cm),
      P(f"<b>In simple words:</b> the model got about {acc:.0f}% of the 39 test tickets right (35 of 39). Rows of the confusion "
        "matrix are the real category and columns the predicted one. Account Update, Application Error, Billing and Report tickets "
        "were all found correctly. The weakest category is Login Issue: only 3 of 6 were found; the others were predicted as "
        "Application Error, Billing and Report, probably because short sentences such as 'unable to ...' or 'not working' also "
        "occur in other categories. I think the result is satisfactory for a first prototype, but with only 39 test tickets "
        "one ticket changes accuracy by 2.6 points, so the number is approximate. The dataset is also clean and simple, so "
        "real-world accuracy would likely be lower.")]
s += [Paragraph("G. Prediction on New Tickets", H),
      P("These 8 sentences were written after training and do not exist in the dataset."),
      tbl(rows, [6.6 * cm, 3 * cm, 3 * cm, 1.5 * cm, 1.9 * cm]), Spacer(1, 6),
      P("7 of 8 were correct. The wrong one ('Please send the revenue summary for the quarter') uses words that never appear in "
        "my Report tickets ('revenue summary'), and the model correctly showed low confidence (38%). This is a useful sign: "
        "low-confidence tickets can be sent to a human. Screenshot of the web app:"),
      img("web_app.png", 8 * cm)]
s += [Paragraph("Bonus: Suggested Response", H),
      P("A rule-based reply is shown for each predicted category (src/responses.py). No external API or key is used.")]
s += [Paragraph("H. Limitations", H)] + bullets([
    "Small, synthetic dataset (191 tickets) written by one person; real tickets are messier, longer and have spelling mistakes.",
    "Only 6 categories; a ticket with two problems gets just one label. The model cannot say 'other/unknown' except by low confidence.",
    "Words never seen in training (e.g. 'revenue summary') are ignored; TF-IDF does not understand meaning or synonyms.",
    "Similar wording between categories (e.g. 'not working') causes confusion.",
    "Test set has only 39 tickets, so results have high variance. Priority/status are random and not used.",
    "Rule-based responses are generic."])
s += [Paragraph("I. Future Improvements", H)] + bullets([
    "Collect a much larger set of real, labelled tickets and add more categories including 'Other'.",
    "Better preprocessing: spelling correction, lemmatization, handling multiple languages.",
    "Better models: sentence embeddings or a fine-tuned transformer; LLM-based classification and replies with human review.",
    "Use a confidence threshold to send uncertain tickets to a human; collect their corrections as feedback and retrain regularly.",
    "Production: REST API (FastAPI), database storage, monitoring, and use of priority/status/date as extra features."])
s += [Paragraph("Libraries and AI usage", H),
      P("Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit, reportlab. "
        "AI tools: Claude (Anthropic) was used to help generate the code, the synthetic dataset and this documentation; "
        "I reviewed it and can explain it. No API keys are included.")]

SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(s)
print("report.pdf written")
