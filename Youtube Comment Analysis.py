#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
import subprocess
from googleapiclient.discovery import build
import streamlit as st
from collections import Counter
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
API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your own YouTube API Key
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

        # Display Metrics Side by Side
        st.subheader('Sentiment Analysis Overview')
        
        # Creating Columns for Side by Side Display
        col1, col2 = st.columns(2)
        
        # Sentiment Distribution Bar Chart
        with col1:
            st.markdown("### Sentiment Distribution")
            sentiment_counts = df['Sentiment'].value_counts()
            plt.figure(figsize=(5, 4))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
            plt.title("Number of Comments per Sentiment")
            plt.ylabel('Count')
            plt.xlabel('Sentiment')
            for i, v in enumerate(sentiment_counts.values):
                plt.text(i, v + 1, str(v), ha='center')
            st.pyplot(plt)
        
        # Sentiment Split Pie Chart
        with col2:
            st.markdown("### Sentiment Split")
            plt.figure(figsize=(5, 5))
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99', '#ff9999'])
            st.pyplot(plt)

        # Top 10 Positive and Negative Comments in Two Columns within a White Box
        st.subheader('Top 10 Positive and Negative Comments')
        
        # Custom CSS for styling
        st.markdown("""
            <style>
            .comment-container {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .comment-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333333;
            }
            .comment-box {
                background-color: #f9f9f9;
                color: #000000;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                border: 1px solid #eee;
                font-size: 14px;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Main container for white box
        st.markdown('<div class="comment-container">', unsafe_allow_html=True)
        
        # Creating Columns inside the white box
        col1, col2 = st.columns(2)
        
        # Top 10 Positive Comments
        with col1:
            st.markdown('<div class="comment-title">Top 10 Positive Comments</div>', unsafe_allow_html=True)
            for comment in df[df['Sentiment'] == 'Positive']['Comment'].head(10):
                st.markdown(f'<div class="comment-box">- {comment}</div>', unsafe_allow_html=True)
        
        # Top 10 Negative Comments
        with col2:
            st.markdown('<div class="comment-title">Top 10 Negative Comments</div>', unsafe_allow_html=True)
            for comment in df[df['Sentiment'] == 'Negative']['Comment'].head(10):
                st.markdown(f'<div class="comment-box">- {comment}</div>', unsafe_allow_html=True)
        
        # Closing the main container
        st.markdown('</div>', unsafe_allow_html=True)


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

        # Confusion Matrix
        st.subheader('Confusion Matrix')
        y_pred = df['Sentiment']
        y_true = ['Positive' if i % 2 == 0 else 'Negative' for i in range(len(df))]
        cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
        st.pyplot(plt)

# WordClouds Side by Side
        st.subheader('Word Clouds of Positive and Negative Comments')
        
        # Creating Columns for Side by Side Display
        col1, col2 = st.columns(2)

        # Positive Word Cloud
        with col1:
            st.markdown("### Positive Comments")
            positive_words = ' '.join(df[df['Sentiment'] == 'Positive']['Processed_Comment'])
            wordcloud = WordCloud(width=600, height=400).generate(positive_words)
            plt.figure(figsize=(6, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        # Negative Word Cloud
        with col2:
            st.markdown("### Negative Comments")
            negative_words = ' '.join(df[df['Sentiment'] == 'Negative']['Processed_Comment'])
            wordcloud = WordCloud(width=600, height=400).generate(negative_words)
            plt.figure(figsize=(6, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

    else:
        st.error('Invalid YouTube URL')
