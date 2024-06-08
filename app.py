import pickle
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer as TV

nltk.download("stopwords")

port_stem = PorterStemmer()

stop_words =set(stopwords.words("english"))
model = pickle.load(open("src/model.pkl","rb"))
def stemming(content):
  stem_con = re.sub('[^a-zA-Z]',' ', content)
  stem_con = stem_con.lower()
  stem_con = stem_con.split()
  stem_con = [port_stem.stem(word) for word in stem_con if word not in stop_words ]
  stem_con = " ".join(stem_con)

  return stem_con

st.title("Sentweet - Twitter Sentiment Analysis")
option = st.selectbox(
   "How do you want to analyse?",
   ("Text Input", "CSV File Input"),
   index=None,
   placeholder="Select input method...",
)

st.write("You selected:", option)

if option=="Text Input":
    text = st.text_input("Enter Tweet", "")
    text=stemming(text)
    X=vectorizer.fit_transform(text)
    result=model.predict(X)
    if result==0:
        st.write("The Tweet is Negative")
    else:
        st.write("The Tweet is Positive")

if option=="CSV File Input":
    text = st.file_uploader("Choose a CSV file",)
    df = pd.read_csv(text)

    option = st.selectbox(
   "Select The Text Columns",
   (data.columns.values),
   index=None,
   placeholder="Select column.",
)

    val=stemming(df[option])
    X=vectorizer.fit_transform(val)
    result=model.predict(X)
    df['result']=result.replace([1,0],["Postive","Negative"])
    st.dataframe(df)


