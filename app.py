from flask import Flask, render_template, request
import pandas as pd
from googleapiclient.discovery import build
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

app = Flask(__name__)

# Your API key
api_key = 'AIzaSyD9_6gPUvffWe7ZlZrCdkJsBSb4MfaYTdw'

def extract_video_id(youtube_url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(1)
    return None

def video_comments(video_id):
    replies = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_response = youtube.commentThreads().list(part='snippet,replies', videoId=video_id).execute()

    while video_response:
        for item in video_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            replies.append(comment)

            if 'replies' in item:
                for reply in item['replies']['comments']:
                    repl = reply['snippet']['textDisplay']
                    replies.append(repl)

        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                pageToken=video_response['nextPageToken'],
                videoId=video_id
            ).execute()
        else:
            break
    return replies

def load_lexicon(file_path):
    lexicon = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                word, score = line.strip().split('\t')
                lexicon[word] = int(score)
            except ValueError:
                continue
    return lexicon

positive_lexicon = load_lexicon('positive.tsv')
negative_lexicon = load_lexicon('negative.tsv')
all_lexicon = {**positive_lexicon, **negative_lexicon}

def classify_sentiment(score):
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    else:
        return "Netral"

def calculate_polarity(words, lexicon):
    total_score = 0
    for word in words:
        total_score += lexicon.get(word, 0)
    return total_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        video_id = extract_video_id(youtube_url)

        if not video_id:
            return render_template('index.html', error="Invalid YouTube URL. Please try again.")

        try:
            comments = video_comments(video_id)
            df = pd.DataFrame(comments, columns=['Comment'])

            # Preprocessing dan klasifikasi sentimen
            df['processed_text'] = df['Comment'].astype(str).str.lower()
            df['polarity_score'] = df['processed_text'].apply(lambda x: calculate_polarity(x.split(), all_lexicon))
            df['sentiment'] = df['polarity_score'].apply(classify_sentiment)

            # Hitung jumlah komentar berdasarkan sentimen
            sentiment_counts = df['sentiment'].value_counts()
            total_positive = sentiment_counts.get('Positif', 0)
            total_negative = sentiment_counts.get('Negatif', 0)
            total_neutral = sentiment_counts.get('Netral', 0)

            # Analisis frekuensi kata
            all_words = ' '.join(df['processed_text']).split()
            word_freq = Counter(all_words).most_common(10)  # Ambil hanya 10 kata populer
            labels, counts = zip(*word_freq)
            total_count = sum(counts)
            counts_percentage = [(count / total_count) * 100 for count in counts]

            # Diagram lingkaran
            plt.figure(figsize=(8, 8))
            plt.pie(counts_percentage, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("hsv", len(labels)))
            plt.title('Frekuensi Kata Populer (Top 10)')
            pie_chart_path = 'static/pie_chart.png'
            plt.savefig(pie_chart_path)
            plt.close()

            # Word Cloud berbentuk logo YouTube
            youtube_logo = np.array(Image.open('static/youtube_logo.png'))
            wordcloud = WordCloud(width=800, height=400, background_color='white', mask=youtube_logo, contour_color='red', contour_width=2).generate(' '.join(all_words))
            wordcloud_path = 'static/wordcloud.png'
            wordcloud.to_file(wordcloud_path)

            # Visualisasi distribusi sentimen
            plt.figure(figsize=(8, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
            plt.title('Distribusi Sentimen Komentar YouTube')
            plt.xlabel('Sentimen')
            plt.ylabel('Jumlah')
            sentiment_path = 'static/sentiment_distribution.png'
            plt.savefig(sentiment_path)
            plt.close()

            return render_template('index.html',
                                   total_positive=total_positive,
                                   total_negative=total_negative,
                                   total_neutral=total_neutral,
                                   word_freq=word_freq,
                                   sentiment_image=sentiment_path,
                                   pie_chart_image=pie_chart_path,
                                   wordcloud_image=wordcloud_path)
        except Exception as e:
            return render_template('index.html', error=f"Error: {e}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
