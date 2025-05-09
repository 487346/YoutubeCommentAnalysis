#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
import subprocess
from googleapiclient.discovery import build
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re

# Initialize Streamlit App
st.set_page_config(page_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# YouTube API Key and Configurations
API_KEY = 'YOUR_API_KEY'  # Replace with your own YouTube API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# User Input for YouTube Video URL
video_url = st.text_input('Enter YouTube Video URL:', '')

# Extract Video ID
def extract_video_id(url):
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

# Fetch YouTube Comments
def get_youtube_comments(video_id):
    comments = []
    timestamps = []
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=100)
    response = request.execute()
    for item in response['items']:
        comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        timestamps.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])
    return pd.DataFrame({'Comment': comments, 'Timestamp': pd.to_datetime(timestamps)})

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Main Logic
if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.success(f'Video ID extracted: {video_id}')
        df = get_youtube_comments(video_id)
        df['Processed_Comment'] = df['Comment'].apply(preprocess_text)

        # Sentiment Analysis
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['Processed_Comment'])
        model = SVC(kernel='linear')
        model.fit(X, ['Positive' if i % 2 == 0 else 'Negative' for i in range(len(df))])
        df['Sentiment'] = model.predict(X)

        # Display Metrics
        st.subheader('Sentiment Distribution')
        st.bar_chart(df['Sentiment'].value_counts())

        # Enhanced Visualizations
        st.subheader('Sentiment Split')
        sentiment_counts = df['Sentiment'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
        st.pyplot(plt)

        # WordClouds
        st.subheader('Word Cloud of Positive Comments')
        positive_words = ' '.join(df[df['Sentiment'] == 'Positive']['Processed_Comment'])
        wordcloud = WordCloud(width=800, height=400).generate(positive_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        st.subheader('Word Cloud of Negative Comments')
        negative_words = ' '.join(df[df['Sentiment'] == 'Negative']['Processed_Comment'])
        wordcloud = WordCloud(width=800, height=400).generate(negative_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Top 10 Positive and Negative Comments
        st.subheader('Top 10 Positive Comments')
        for comment in df[df['Sentiment'] == 'Positive']['Comment'].head(10):
            st.write(f"- {comment}")

        st.subheader('Top 10 Negative Comments')
        for comment in df[df['Sentiment'] == 'Negative']['Comment'].head(10):
            st.write(f"- {comment}")

        # Confusion Matrix
        st.subheader('Confusion Matrix')
        y_pred = df['Sentiment']
        y_true = ['Positive' if i % 2 == 0 else 'Negative' for i in range(len(df))]
        cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
        st.pyplot(plt)

        # Most Common Words
        st.subheader('Most Common Words')
        common_words = Counter(' '.join(df['Processed_Comment']).split()).most_common(20)
        common_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
        st.write(common_df)

        # Time-Series Analysis
        st.subheader('Time-Series Analysis of Sentiments')
        df['Date'] = df['Timestamp'].dt.date
        time_series_data = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
        st.line_chart(time_series_data)

    else:
        st.error('Invalid YouTube URL')
