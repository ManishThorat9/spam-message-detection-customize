import random

import pandas as pd
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

dataset = pd.read_csv('spam.csv', encoding='ISO-8859-15', header=0)
spam = dataset[dataset['v1'] == 'spam']
ham = dataset[dataset['v1'] == 'ham']


# returns a random message from dataset with 50% chance of spam
def get_random_message():
    if random.randint(0, 1) == 1:
        return spam['v2'].iloc[random.randint(0, len(spam) - 1)]
    else:
        return ham['v2'].iloc[random.randint(0, len(ham) - 1)]


def transform_text(text):
    # convert in lower case
    text = text.lower()
    # convert into list
    text = nltk.word_tokenize(text)
    # remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.text("=" * 80)

st.title("Random Message spam Classifier")

input_sms = get_random_message()
st.text(input_sms)

if st.button(' Predict '):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

st.text("=" * 80)
st.title("Customise Message spam Classifier ")

input_sms1 = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms1)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
st.text("=" * 80)
# 2962