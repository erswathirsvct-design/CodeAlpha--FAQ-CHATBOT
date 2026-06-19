import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')
faq = pd.read_csv("faq.csv")
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]

    return " ".join(tokens)
faq['Processed'] = faq['Question'].apply(preprocess)
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq['Processed'])
def chatbot_response(user_input):
    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.2:
        return faq.iloc[best_match]['Answer']
    else:
        return "Sorry, I couldn't understand your question."
st.title("FAQ Chatbot ")
user_question = st.text_input("Ask your question here:")
if st.button("Send"):
    response = chatbot_response(user_question)
    st.success(response)