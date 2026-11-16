"""Simple web interface (Task 6 Option B) using Streamlit.  Run:  streamlit run src/app.py"""
import streamlit as st
from predict import load_model, predict
from responses import suggest_response


@st.cache_resource
def get_model():
    return load_model()


st.set_page_config(page_title="Customer Support Classifier", page_icon="🎫")
st.title("🎫 Customer Support Classifier")
text = st.text_area("Ticket Description:", placeholder="e.g. The application is very slow today.", height=120)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter a ticket description.")
    else:
        cat, conf, probs = predict(get_model(), text)
        st.subheader("Predicted Category")
        st.success(cat)
        st.metric("Confidence", f"{conf:.1f}%")
        if conf < 50:
            st.info("Low confidence - this ticket may need a human to review it.")
        st.subheader("Suggested Response")
        st.write(suggest_response(cat))
        st.bar_chart({k: v * 100 for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}, horizontal=True)
