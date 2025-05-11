# ---- Imports ----
#!/usr/bin/env python
# coding: utf-8

import sys
import subprocess
from googleapiclient.discovery import build
import streamlit as st
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

nltk.download('stopwords')

# ---- Streamlit App Configurations ----
st.set_page_config(page_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# ---- YouTube API Key and Configurations ----
API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your YouTube API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# ---- User Input for YouTube Video URL ----
video_url = st.text_input('Enter YouTube Video URL:', '')

# ---- Extract Video ID from URL ----
def extract_video_id(url):
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

# ---- Fetch YouTube Comments ----
def get_youtube_comments(video_id):
    comments, timestamps, users, likes = [], [], [], []
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=100)
    response = request.execute()
    for item in response['items']:
        comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        timestamps.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])
        users.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])
        likes.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
    return pd.DataFrame({
        'User': users,
        'Comment': comments,
        'Timestamp': pd.to_datetime(timestamps),
        'Likes': likes
    })

# ---- Data Preprocessing ----
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# ---- Sentiment Analysis using VADER ----
def sentiment_analysis(comments):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for comment in comments:
        vs = analyzer.polarity_scores(comment)
        if vs['compound'] > 0.05:
            sentiments.append('Positive')
        elif vs['compound'] < -0.05:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
    return sentiments

# ---- Pre-trained Spam Detection Model Loading ----
@st.cache_resource
def load_spam_model():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("spam_detector_model.pkl")
    except:
        # Sample Spam Dataset
        data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv", sep='\t', header=None)
        data.columns = ['Label', 'Message']

        # Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data['Message'])
        y = data['Label'].map({'ham': 0, 'spam': 1})

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model Training
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Saving the model for future use
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        joblib.dump(model, "spam_detector_model.pkl")

    return vectorizer, model

# 🚀 Load the pre-trained model outside the if condition
tfidf_vectorizer, spam_detector_model = load_spam_model()

# ---- Enhanced Spam Detection ----
def detect_spam(comment):
    """ Use pre-trained model to detect spam """
    comment_transformed = tfidf_vectorizer.transform([comment])
    prediction = spam_detector_model.predict(comment_transformed)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# ---- Main Logic ----
if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.success(f'Video ID extracted: {video_id}')
        df = get_youtube_comments(video_id)
        df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
        df['Sentiment'] = sentiment_analysis(df['Processed_Comment'])
        df['Spam'] = df['Comment'].apply(detect_spam)

        # ---- Time-Series Analysis ----
        st.subheader('⏳ Time-Series Analysis of Sentiments')
        df['Date'] = df['Timestamp'].dt.date
        time_series_data = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
        st.line_chart(time_series_data)

        # ---- Spam Detection & Analysis ----
        st.subheader('🚫 Spam Detection & Analysis')
        spam_counts = df['Spam'].value_counts()
        
        # Visualization
        fig, ax = plt.subplots()
        sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Reds')
        plt.title("Spam Detection Overview")
        plt.ylabel('Number of Comments')
        plt.xlabel('Comment Type')
        st.pyplot(fig)

        # Display Spam Comments
        spam_comments = df[df['Spam'] == 'Spam']
        if not spam_comments.empty:
            st.markdown("### 🚩 Detected Spam Comments and Usernames")
            st.dataframe(spam_comments[['User', 'Comment']])
        else:
            st.success("No spam comments detected! 🎉")

    else:
        st.error('Invalid YouTube URL')
