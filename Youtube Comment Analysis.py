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
import joblib
import os
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Initialize Streamlit App
st.set_page_config(page_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# YouTube API Key and Configurations
API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your own YouTube API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# User Input for YouTube Video URL
video_url = st.text_input('Enter YouTube Video URL:', '')

# Button to fetch comments
if st.button("Fetch Comments"):
    if video_url:  # Only proceed if there is input
        video_id = extract_video_id(video_url)
        if video_id:
            st.success(f'Video ID extracted: {video_id}')
            # Proceed with fetching comments
            df = get_youtube_comments(video_id)
            if not df.empty:
                st.success("Comments fetched successfully!")
            else:
                st.error("No comments found for this video.")
        else:
            st.error('Invalid YouTube URL')
    else:
        st.warning('Please enter a YouTube URL.')

# Extract Video ID
def extract_video_id(url):
    print(f"URL entered: {url}")  # Debugging line
    patterns = [
        r'v=([0-9A-Za-z_-]{11})',
        r'\/([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            print(f"Video ID extracted: {video_id}")
            return video_id
    print("Video ID extraction failed.")  # Debugging line
    return None

# Fetch YouTube Comments with Username
def get_youtube_comments(video_id):
    comments = []
    timestamps = []
    users = []
    request = youtube.commentThreads().list(
        part='snippet', 
        videoId=video_id, 
        textFormat='plainText', 
        maxResults=100
    )
    
    response = request.execute()
    
    for item in response['items']:
        comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        timestamps.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])
        users.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])  # Fetch username
    
    return pd.DataFrame({'User': users, 'Comment': comments, 'Timestamp': pd.to_datetime(timestamps)})

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
            
        # Initialize VADER
        sia = SentimentIntensityAnalyzer()
        
        # Function to determine sentiment
        def detect_sentiment(comment):
            score = sia.polarity_scores(comment)
            if score['compound'] > 0.05:
                return 'Positive'
            elif score['compound'] < -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        # Apply sentiment detection to each comment if df exists
        if 'df' in locals():
            df['Sentiment'] = df['Processed_Comment'].apply(detect_sentiment)
        
        # Initialize VADER
        sia = SentimentIntensityAnalyzer()
        
        # Function to determine sentiment
        def detect_sentiment(comment):
            score = sia.polarity_scores(comment)
            if score['compound'] > 0.05:
                return 'Positive'
            elif score['compound'] < -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        # Apply sentiment detection to each comment if df exists
        if 'df' in locals():
            df['Sentiment'] = df['Processed_Comment'].apply(detect_sentiment)
        
        # Display sentiment distribution
        st.subheader('Sentiment Analysis Overview')
        sentiment_counts = df['Sentiment'].value_counts()
        
        # Creating Columns for Side by Side Display
        col1, col2 = st.columns(2)
        
        # Sentiment Distribution Bar Chart
        with col1:
            plt.figure(figsize=(4, 4))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
            plt.title("Number of Comments per Sentiment")
            plt.ylabel('Count')
            plt.xlabel('Sentiment')
            st.pyplot(plt)
        
        # Sentiment Split Pie Chart
        with col2:
            plt.figure(figsize=(4, 4))  # Changed to square size for better look
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                    colors=['#66b3ff', '#99ff99', '#ff9999'])
            plt.gca().set_aspect('equal')  # This keeps the pie chart circular
            st.pyplot(plt)
        
        # Creating Columns for Side by Side Display
        col1, col2 = st.columns(2)
        
        # Top 10 Positive Comments
        with col1:
            st.markdown("### Top 10 Positive Comments")
            for comment in df[df['Sentiment'] == 'Positive']['Comment'].head(10):
                st.write(f"- {comment}")
        
        # Top 10 Negative Comments
        with col2:
            st.markdown("### Top 10 Negative Comments")
            for comment in df[df['Sentiment'] == 'Negative']['Comment'].head(10):
                st.write(f"- {comment}")


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

        # Assuming `df` is already available
        y_pred = df['Sentiment']  # assuming this column contains the sentiment predictions
        y_true = ['Positive' if i % 3 == 0 else 'Negative' if i % 3 == 1 else 'Neutral' for i in range(len(df))]  # Placeholder
        
        # Generate the confusion matrix with all three classes
        cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral'])
        
        # Create two columns for display
        col1, col2 = st.columns(2)
        
        # Display Confusion Matrix
        with col1:
            st.subheader('Confusion Matrix')
        
        # Plot the Confusion Matrix in the second column
        with col2:
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Positive', 'Negative', 'Neutral'], 
                        yticklabels=['Positive', 'Negative', 'Neutral'])
            plt.title('Confusion Matrix')
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
            
# ---- Pre-trained Spam Detection Model Loading ----
@st.cache_resource
def load_spam_model():
    """
    Load the pre-trained spam detection model and vectorizer.
    If not available, train on sample data, save, and load it.
    """
    try:
        # Check if the files exist in the current directory
        if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("spam_detector_model.pkl"):
            st.info("Loading pre-trained models...")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            model = joblib.load("spam_detector_model.pkl")
        else:
            st.warning("Pre-trained models not found. Training a new model...")

            # ---- Sample Spam Dataset ----
            data = pd.read_csv(
                "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv",
                sep='\t',
                header=None
            )
            data.columns = ['Label', 'Message']

            # ---- Vectorization ----
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(data['Message'])
            y = data['Label'].map({'ham': 0, 'spam': 1})

            # ---- Train-Test Split ----
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # ---- Model Training ----
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB()
            model.fit(X_train, y_train)

            # ---- Saving the model for future use ----
            joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
            joblib.dump(model, "spam_detector_model.pkl")

            # ---- Model Evaluation ----
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.success(f"Spam Detection Model trained with an accuracy of: {accuracy:.2f}")

    except Exception as e:
        st.error(f"🚫 Error loading models: {e}")
        return None, None

    return vectorizer, model
# 🚀 Load the pre-trained model outside the main logic to optimize performance
tfidf_vectorizer, spam_detector_model = load_spam_model()


# ---- Enhanced Spam Detection ----
def detect_spam(comment):
    """
    Use the pre-trained model to detect if a comment is spam or not.
    """
    comment_transformed = tfidf_vectorizer.transform([comment])
    prediction = spam_detector_model.predict(comment_transformed)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'


# ---- Apply the new spam detection on YouTube comments ----
if 'df' in locals():
    df['Spam'] = df['Comment'].apply(detect_spam)

    # ---- Display Spam Comments and Analysis ----
    st.subheader('🚫 Spam Detection & Analysis')
    
    # ---- Display Spam Comments ----
    spam_comments = df[df['Spam'] == 'Spam']
    if not spam_comments.empty:
        st.markdown("### 🚩 Detected Spam Comments and Usernames")
        st.dataframe(spam_comments[['User', 'Comment']])

        # ---- Top Spam Commenters ----
        st.markdown("### 🏆 Top Spam Commenters")
        top_spammers = spam_comments['User'].value_counts().head(10).reset_index()
        top_spammers.columns = ['Username', 'Spam Count']
        st.write(top_spammers)
    else:
        st.success("No spam comments detected! 🎉")
        
    # ---- Visualization ----
    spam_counts = df['Spam'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Reds')
    plt.title("Spam Detection Overview")
    plt.ylabel('Number of Comments')
    plt.xlabel('Comment Type')
    st.pyplot(fig)
    
else:
    st.error('Invalid YouTube URL')
