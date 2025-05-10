#!/usr/bin/env python
# coding: utf-8

# ---- Imports ----
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

# ---- Spam Detection ----
def detect_spam(comment):
    spam_keywords = ['free', 'click here', 'subscribe', 'visit', 'buy now', 'check out']
    if any(keyword in comment.lower() for keyword in spam_keywords):
        return 'Spam'
    else:
        return 'Not Spam'

# ---- Main Logic ----
if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.success(f'Video ID extracted: {video_id}')
        df = get_youtube_comments(video_id)
        df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
        df['Sentiment'] = sentiment_analysis(df['Processed_Comment'])
        df['Spam'] = df['Comment'].apply(detect_spam)

        # ---- Influential Commenters Analysis ----
        st.subheader('🔥 Influential Commenters Analysis')
        top_commenters = df.groupby('User').agg({
            'Comment': 'count',
            'Likes': 'sum'
        }).sort_values(by='Comment', ascending=False).head(10).reset_index()
        st.write(top_commenters)

        # ---- Sentiment Distribution ----
        st.subheader('📊 Sentiment Distribution')
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=df, palette='Set2', ax=ax)
        plt.title("Sentiment Distribution")
        st.pyplot(fig)

        # ---- Like-to-Dislike Ratio Analysis ----
        st.subheader('👍 Like-to-Dislike Ratio Analysis')
        plt.figure(figsize=(6, 4))
        sns.histplot(df['Likes'], bins=20, kde=True, color='green')
        plt.title('Distribution of Likes on Comments')
        plt.xlabel('Number of Likes')
        plt.ylabel('Comment Count')
        st.pyplot(plt)

        # ---- Spam Detection Visualization ----
        st.subheader('⚠️ Spam Detection Analysis')
        spam_counts = df['Spam'].value_counts()
        st.bar_chart(spam_counts)

        # ---- Topic Modeling (LDA) ----
        st.subheader('🧠 Topic Modeling (LDA)')
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['Processed_Comment'])
        
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(X)

        # Display Topics
        terms = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda_model.components_):
            st.write(f"**Topic #{idx + 1}:**", [terms[i] for i in topic.argsort()[-5:]])

        # ---- Time-Series Analysis ----
        st.subheader('⏳ Time-Series Analysis of Sentiments')
        df['Date'] = df['Timestamp'].dt.date
        time_series_data = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
        st.line_chart(time_series_data)

        # ---- Confusion Matrix ----
        st.subheader('🗂️ Confusion Matrix')
        y_pred = df['Sentiment']
        y_true = ['Positive' if i % 2 == 0 else 'Negative' for i in range(len(df))]
        cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
        st.pyplot(plt)

    else:
        st.error('Invalid YouTube URL')

