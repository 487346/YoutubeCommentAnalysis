#!/usr/bin/env python

# coding: utf-8

# In\[ ]:

import sys
import subprocess
from googleapiclient.discovery import build
import streamlit as st
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature\_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy\_score, classification\_report, confusion\_matrix
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import joblib
import os
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader\_lexicon')

# Initialize Streamlit App

st.set\_page\_config(page\_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# YouTube API Key and Configurations

API\_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your own YouTube API Key
youtube = build('youtube', 'v3', developerKey=API\_KEY)

# User Input for YouTube Video URL

video\_url = st.text\_input('Enter YouTube Video URL:', '')

# Button to fetch comments

if st.button("Fetch Comments"):
if video\_url:  # Only proceed if there is input
video\_id = extract\_video\_id(video\_url)
if video\_id:
st.success(f'Video ID extracted: {video\_id}')
\# Proceed with fetching comments
df = get\_youtube\_comments(video\_id)
if not df.empty:
st.success("Comments fetched successfully!")
else:
st.error("No comments found for this video.")
else:
st.error('Invalid YouTube URL')
else:
st.warning('Please enter a YouTube URL.')

# Extract Video ID

def extract\_video\_id(url):
print(f"URL entered: {url}")  # Debugging line
patterns = \[
r'v=(\[0-9A-Za-z\_-]{11})',
r'/(\[0-9A-Za-z\_-]{11})',
r'youtu.be/(\[0-9A-Za-z\_-]{11})',
r'embed/(\[0-9A-Za-z\_-]{11})'
]
for pattern in patterns:
match = re.search(pattern, url)
if match:
video\_id = match.group(1)
print(f"Video ID extracted: {video\_id}")
return video\_id
print("Video ID extraction failed.")  # Debugging line
return None

# Fetch YouTube Comments with Username

def get\_youtube\_comments(video\_id):
comments = \[]
timestamps = \[]
users = \[]
request = youtube.commentThreads().list(
part='snippet',
videoId=video\_id,
textFormat='plainText',
maxResults=100
)

```
response = request.execute()

for item in response['items']:
    comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    timestamps.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])
    users.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])  # Fetch username

return pd.DataFrame({'User': users, 'Comment': comments, 'Timestamp': pd.to_datetime(timestamps)})
```

# Data Preprocessing

def preprocess\_text(text):
text = text.lower()
text = re.sub(r'http\S+', '', text)
text = re.sub(r'\[^a-zA-Z\s]', '', text)
text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
return text

# Main Logic

if video\_url:
video\_id = extract\_video\_id(video\_url)
if video\_id:
st.success(f'Video ID extracted: {video\_id}')
df = get\_youtube\_comments(video\_id)
df\['Processed\_Comment'] = df\['Comment'].apply(preprocess\_text)

```
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
    total = cm.sum() if cm.size > 0 else 1
    
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
    
    # Calculate TP, FP, FN, TN for each class
    TP = [cm[i, i] for i in range(3)]
    FP = [cm[:, i].sum() - cm[i, i] for i in range(3)]
    FN = [cm[i, :].sum() - cm[i, i] for i in range(3)]
    TN = [total - (TP[i] + FP[i] + FN[i]) for i in range(3)]
    
    # Calculate percentages
    TP_pct = [round((x / total) * 100, 2) for x in TP]
    FP_pct = [round((x / total) * 100, 2) for x in FP]
    FN_pct = [round((x / total) * 100, 2) for x in FN]
    TN_pct = [round((x / total) * 100, 2) for x in TN]
    
    metrics_data = {
        'Class': ['Positive', 'Negative', 'Neutral'],
        'True Positive (TP)': TP,
        'False Positive (FP)': FP,
        'False Negative (FN)': FN,
        'True Negative (TN)': TN,
        'TP %': TP_pct,
        'FP %': FP_pct,
        'FN %': FN_pct,
        'TN %': TN_pct
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot the metrics with percentage display
    with col2:
        st.subheader('Confusion Matrix Breakdown')
        plt.figure(figsize=(7, 5))
        
        # Plotting each bar separately for better visibility
        bar_width = 0.2
        st.write("Metrics DataFrame Preview:")
        st.write(metrics_df)
        st.write("Length of Classes:", len(metrics_df['Class']))
        positions = np.arange(len(metrics_df['Class'])) if len(metrics_df) > 0 else np.array([0])
        
        # Bars
        plt.bar(positions, metrics_df['True Positive (TP)'], width=bar_width, label='TP', color='green')
        plt.bar(positions + bar_width, metrics_df['False Positive (FP)'], width=bar_width, label='FP', color='red')
        plt.bar(positions + bar_width * 2, metrics_df['False Negative (FN)'], width=bar_width, label='FN', color='orange')
        plt.bar(positions + bar_width * 3, metrics_df['True Negative (TN)'], width=bar_width, label='TN', color='blue')
        
        # Display percentage on top of each bar
        for i, pos in enumerate(positions):
            plt.text(pos, metrics_df['True Positive (TP)'][i] + 1, f"{metrics_df['TP %'][i]}%", ha='center')
            plt.text(pos + bar_width, metrics_df['False Positive (FP)'][i] + 1, f"{metrics_df['FP %'][i]}%", ha='center')
            plt.text(pos + bar_width * 2, metrics_df['False Negative (FN)'][i] + 1, f"{metrics_df['FN %'][i]}%", ha='center')
            plt.text(pos + bar_width * 3, metrics_df['True Negative (TN)'][i] + 1, f"{metrics_df['TN %'][i]}%", ha='center')
        
        plt.xticks(positions + bar_width * 1.5, metrics_df['Class'])
        plt.legend()
        plt.title('TP, FP, FN, TN Breakdown by Class')
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
        
```

# ---- Pre-trained Spam Detection Model Loading ----

@st.cache\_resource
def load\_spam\_model():
"""
Load the pre-trained spam detection model and vectorizer.
If not available, train on sample data, save, and load it.
"""
try:
\# Check if the files exist in the current directory
if os.path.exists("tfidf\_vectorizer.pkl") and os.path.exists("spam\_detector\_model.pkl"):
st.info("Loading pre-trained models...")
vectorizer = joblib.load("tfidf\_vectorizer.pkl")
model = joblib.load("spam\_detector\_model.pkl")
else:
st.warning("Pre-trained models not found. Training a new model...")

```
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
```

# 🚀 Load the pre-trained model outside the main logic to optimize performance

tfidf\_vectorizer, spam\_detector\_model = load\_spam\_model()

# ---- Enhanced Spam Detection ----

def detect\_spam(comment):
"""
Use the pre-trained model to detect if a comment is spam or not.
"""
comment\_transformed = tfidf\_vectorizer.transform(\[comment])
prediction = spam\_detector\_model.predict(comment\_transformed)
return 'Spam' if prediction\[0] == 1 else 'Not Spam'

# ---- Apply the new spam detection on YouTube comments ----

if 'df' in locals():
df\['Spam'] = df\['Comment'].apply(detect\_spam)

```
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
```

else:
st.error('Invalid YouTube URL')
