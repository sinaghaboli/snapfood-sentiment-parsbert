"""
اپ Streamlit برای تحلیل احساسات کامنت‌های فارسی Snapfood با مدل ParsBERT Fine-tuned.
این فایل را app.py نام‌گذاری کن و در ریشه‌ی ریپازیتوری گیت‌هابت قرار بده.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="تحلیل احساسات Snapfood - ParsBERT",
    page_icon="🍔",
    layout="centered"
)

st.markdown(
    """
    <style>
    * {
        direction: rtl !important;
        text-align: right !important;
    }
    .stTextArea textarea {
        direction: rtl !important;
        text-align: right !important;
    }
    .stButton button {
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        direction: ltr !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_REPO_ID = "Capblack/snapfood-parsbert-sentiment"

LABEL_MAP = {0: "😊 HAPPY (مثبت)", 1: "😞 SAD (منفی)"}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO_ID)
    model.eval()
    return tokenizer, model


def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())
    return LABEL_MAP[pred_id], float(probs[pred_id].item()), {
        "HAPPY": float(probs[0].item()),
        "SAD": float(probs[1].item()),
    }


st.title("🍔 تحلیل احساسات کامنت‌های Snapfood")
st.markdown("### مدل: ParsBERT Fine-tuned | دقت روی Test: 87.61%")
st.write("یک کامنت فارسی وارد کنید تا مدل احساس آن را (مثبت یا منفی) تشخیص دهد.")

with st.spinner("در حال بارگذاری مدل..."):
    tokenizer, model = load_model()

text_input = st.text_area(
    "متن کامنت را وارد کنید",
    placeholder="مثلاً: غذا خیلی خوشمزه بود و به موقع رسید...",
    height=100
)

examples = [
    "غذا خیلی خوشمزه بود و به موقع رسید، عالی بود!",
    "غذا سرد بود و پیک خیلی دیر رسید، اصلا راضی نبودم.",
    "غذا خوب بود ولی بسته‌بندی بد بود و کمی هم دیر رسید.",
]

st.write("نمونه‌های آماده:")
cols = st.columns(3)
for i, ex in enumerate(examples):
    if cols[i].button(f"نمونه {i+1}", key=f"ex_{i}"):
        st.session_state["text_input_value"] = ex

if "text_input_value" in st.session_state:
    text_input = st.session_state["text_input_value"]

analyze = st.button("تحلیل کن 🔍", type="primary")

if analyze:
    if not text_input or not text_input.strip():
        st.warning("⚠️ لطفاً یک متن وارد کنید.")
    else:
        label, confidence, prob_dict = predict_sentiment(text_input, tokenizer, model)
        st.markdown(f"## نتیجه: {label}")
        st.progress(confidence)
        st.write(f"**درصد اطمینان: {confidence:.2%}**")

        col1, col2 = st.columns(2)
        col1.metric("HAPPY (مثبت)", f"{prob_dict['HAPPY']:.1%}")
        col2.metric("SAD (منفی)", f"{prob_dict['SAD']:.1%}")

st.markdown("---")
st.caption("پروژه دانشگاهی تحلیل احساسات فارسی | مدل ParsBERT Fine-tuned روی دیتاست Snapfood")
