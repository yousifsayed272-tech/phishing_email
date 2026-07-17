import streamlit as st
import pickle
from tensorflow import keras
import re
import Levenshtein

model = keras.models.load_model("phishing_model.h5")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

KEYWORDS = ["urgent", "verify", "password", "bank", "click", "login", "account"]

TRUSTED_DOMAINS = [
    "paypal.com",
    "google.com",
    "amazon.com",
    "facebook.com",
    "apple.com",
    "microsoft.com"
]

def rule_based_analysis(text):
    score = 0
    reasons = []
    text_lower = text.lower()

    for word in KEYWORDS:
        if word in text_lower:
            score += 1
            reasons.append(f"⚠️ Suspicious keyword: {word}")

    if "http" in text_lower or "www" in text_lower:
        score += 2
        reasons.append("🔗 Contains link")

    return score, reasons

def check_domain(sender):
    if not sender or "@" not in sender:
        return 0, []

    domain = sender.split("@")[-1]

    if domain in TRUSTED_DOMAINS:
        return 0, [f"✅ Trusted domain: {domain}"]
    else:
        return 1, [f"⚠️ Untrusted domain: {domain}"]

def check_domain_similarity(sender):
    reasons = []
    score = 0

    if not sender or "@" not in sender:
        return 0, []

    domain = sender.split("@")[-1]

    for trusted in TRUSTED_DOMAINS:
        similarity = Levenshtein.ratio(domain, trusted)

        if similarity > 0.8 and domain != trusted:
            score += 2
            reasons.append(f"⚠️ Looks like {trusted} (similarity {round(similarity,2)})")

    return score, reasons

def highlight_text(text):
    highlighted = text

    for word in KEYWORDS:
        pattern = re.compile(rf"({word})", re.IGNORECASE)
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)

    return highlighted

def analyze_email(text, sender):
    vec = vectorizer.transform([text]).toarray()
    prob = model.predict(vec)[0][0]

    rule_score, reasons = rule_based_analysis(text)
    domain_score, domain_reason = check_domain(sender)
    sim_score, sim_reason = check_domain_similarity(sender)

    reasons += domain_reason + sim_reason

    risk_score = (prob * 0.7) + (rule_score * 0.1) + (domain_score * 0.05) + (sim_score * 0.05)
    risk_score = round(min(risk_score, 1.0), 3)

    if risk_score >= 0.8:
        label = "🚨 Phishing"
        color = "red"
    elif risk_score >= 0.5:
        label = "⚠️ Suspicious"
        color = "orange"
    else:
        label = "✅ Safe"
        color = "green"

    return label, risk_score, reasons, color

st.set_page_config(page_title="Phishing Detector", page_icon="📧")

st.title("📧 Phishing Email Detector")

sender = st.text_input("📨 Sender Email (optional)")
email_text = st.text_area("✉️ Enter Email Content", height=200)

if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email!")
    else:
        label, risk, reasons, color = analyze_email(email_text, sender)

        st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        st.subheader("📊 Risk Level")
        st.progress(int(risk * 100))
        st.write(f"Risk Score: {risk}")

        st.subheader("🔍 Highlighted Email")
        st.markdown(highlight_text(email_text), unsafe_allow_html=True)

        st.subheader("⚠️ Explanation")
        if reasons:
            for r in reasons:
                st.write("-", r)
        else:
            st.write("No suspicious indicators found.")