
# coding: utf-8

# In[9]:


import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[3]:


nltk.download('vader_lexicon')


# In[5]:


st.title('Chrome reviews')


# In[6]:


uploaded_file = st.file_uploader("Please upload the chrome reviews file")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    df = df[["ID", "Text", "Star", "Thumbs Up"]]
    print("Null data in the train dataset:\n", df.isnull().any())
    df = df.dropna()
    sid = SentimentIntensityAnalyzer()
    df["compound_score"] = [sid.polarity_scores(text)['compound'] for text in df['Text'].values]
    data = df[['Text','Star','compound_score']][((df['Star']<3) & (df['compound_score']>0.35)) | ((df['Star']>3) & (df['compound_score']<-0.05))]
    st.subheader("Chrome reviews where the semantics of review text does not match rating")
    st.write(data)

