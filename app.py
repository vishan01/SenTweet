import pickle
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer as TV

nltk.download("stopwords")

port_stem = PorterStemmer()

stop_words =set(stopwords.words("english"))
vectorizer = TV()
model = pickle.load(open("src/model.pkl","rb"))
vectorizer = pickle.load(open('src/vectorizer.pkl','rb'))
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
    if(text):
        text=stemming(text)
        X=vectorizer.transform([text])
        result=model.predict(X)
        if result==0:
            st.write("The Tweet is Negative")
        else:
            st.write("The Tweet is Positive")

if option=="CSV File Input":
    text = st.file_uploader("Choose a CSV file",)
    if(text):
        df = pd.read_csv(text)

        option = st.selectbox(
    "Select The Text Columns",
    (df.columns.values),
    index=None,
    placeholder="Select column.",
    )
        if(option):
            val=df[option].astype(str).apply(stemming)
            X=vectorizer.transform(val)
            result=model.predict(X)
            val=[]
            for x in result:
                if x==1:
                    val.append("Positive")
                else:
                    val.append("Negative")
            
            df['result']=val
            st.dataframe(df)


