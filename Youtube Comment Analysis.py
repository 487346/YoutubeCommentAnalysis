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
import csv
from io import StringIO

# Initialize Streamlit App
st.set_page_config(page_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# YouTube API Key and Configurations
API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your own YouTube API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# User Input for YouTube Video URL
video_url = st.text_input('Enter YouTube Video URL:', '')

""from flask import Flask, request, send_file
import pandas as pd
from googleapiclient.discovery import build
import os

app = Flask(__name__)

def get_youtube_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    comments_data = {
        'User': [],
        'Comment': [],
        'Timestamp': []
    }
    
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100
    )
    
    while request:
        response = request.execute()
        
        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comments_data['User'].append(snippet['authorDisplayName'])
            comments_data['Comment'].append(snippet['textDisplay'])
            comments_data['Timestamp'].append(snippet['publishedAt'])
        
        request = youtube.commentThreads().list_next(request, response)
    
    df = pd.DataFrame(comments_data)
    csv_path = 'comments.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

@app.route('/download-comments')
def download_comments():
    video_id = request.args.get('videoId')
    if not video_id:
        return 'Video ID is required', 400
    
    csv_path = get_youtube_comments(video_id)
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=5000)

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

# 2️⃣ Fetch YouTube Comments with Timestamps and Usernames
def get_youtube_comments(youtube, video_id):
    all_comments = []  # To store each comment's data

    # Initial API request
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100
    )
    

    # Execute the request and process the first batch of comments
    response = request.execute()
    print("Fetching comments...")

    # Pagination Loop
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
            username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            all_comments.append({
                "Comment": comment,
                "Timestamp": timestamp,
                "User": username
            })

        # Check if there is another page of comments
        if 'nextPageToken' in response:
            next_page_token = response['nextPageToken']
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()
        else:
            print(f"All comments fetched: {len(all_comments)}")
            break

    # Return the complete list of comments
    return all_comments

# 3️⃣ Main Execution Block
if __name__ == "__main__":
    # Provide your YouTube API Key
    API_KEY = 'YOUR_YOUTUBE_API_KEY'  # <-- Replace with your actual API Key
    VIDEO_URL = 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'  # <-- Replace with the actual video URL

    # Extract Video ID from URL
    video_id = VIDEO_URL.split('v=')[1]

    # Initialize YouTube client
    youtube = initialize_youtube_client(API_KEY)

    # Fetch comments, timestamps, and users
    comment_data = get_youtube_comments(youtube, video_id)

    # 4️⃣ Convert to DataFrame
    df = pd.DataFrame(comment_data)

    # 5️⃣ Save to CSV for future analysis
    df.to_csv('youtube_comments.csv', index=False)
    print("\nData saved to youtube_comments.csv")

    # Display DataFrame
    print("\nSample Data:")
    print(df.head())

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
        
        
        # Filter Top 10 Positive and Negative Comments
        top_positive_comments = df[df['Sentiment'] == 'Positive'][['Comment']].head(10).reset_index(drop=True)
        top_negative_comments = df[df['Sentiment'] == 'Negative'][['Comment']].head(10).reset_index(drop=True)
        
        # Assign proper column names
        top_positive_comments.columns = ['Top 10 Positive Comments']
        top_negative_comments.columns = ['Top 10 Negative Comments']
        
        # Layout with two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top 10 Positive Comments")
            st.table(top_positive_comments)
        
        with col2:
            st.markdown("### Top 10 Negative Comments")
            st.table(top_negative_comments)

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
        y_true = ['Positive' if i % 3 == 0 else 'Negative' if i % 3 == 1 else 'Neutral' for i in range(len(df))]  # Placeholder
        y_pred = np.array(df['Sentiment'])
        
        # Generate the confusion matrix with all three classes
        cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral'])
        
        # Extract TN, TP, FP, FN
        TP = cm[1, 1]  # True Positive
        TN = cm[0, 0]  # True Negative
        FP = cm[0, 1]  # False Positive
        FN = cm[1, 0]  # False Negative
        
        # Total count of samples
        total = np.sum(cm)
        
        # Calculate the percentages
        TN_percentage = (TN / total) * 100
        TP_percentage = (TP / total) * 100
        FP_percentage = (FP / total) * 100
        FN_percentage = (FN / total) * 100
        
        # Layout with two columns
        col1, col2 = st.columns(2)
        # Create two columns for display
        col1, col2 = st.columns(2)
        
        # Display Confusion Matrix
        with col1:
            st.subheader('Confusion Matrix')
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Positive', 'Negative', 'Neutral'],
                        yticklabels=['Positive', 'Negative', 'Neutral'])
            plt.title('Confusion Matrix')
            st.pyplot(plt)
        
        # Plot the metrics
        with col2:
            st.subheader('TN, TP, FP, FN Distribution')
                
            # Plotting the values of TN, TP, FP, FN
            labels = ['True Negative (TN)', 'True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)']
            values = [TN, TP, FP, FN]
            percentages = [TN_percentage, TP_percentage, FP_percentage, FN_percentage]
        
            # Plot
            plt.figure(figsize=(5, 4))
            bars = plt.bar(labels, values, color=['green', 'blue', 'red', 'orange'])
        
            # Add percentages on top of each bar
            for bar, percentage in zip(bars, percentages):
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{percentage:.2f}%', ha='center', va='bottom')
        
            plt.title('Confusion Matrix Components')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
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
