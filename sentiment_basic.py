import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load data
data = pd.read_csv("data/reviews.csv")

# Clean data
data = data.dropna(subset=['review'])
data['review'] = data['review'].astype(str)

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.6:
        return "Happy"
    elif polarity > 0.2:
        return "Satisfied"
    elif polarity < -0.6:
        return "Angry"
    elif polarity < -0.2:
        return "Sad"
    else:
        return "Neutral"

# Apply analysis
data['Sentiment'] = data['review'].apply(get_sentiment)
data['Emotion'] = data['review'].apply(get_emotion)

# Save output
data.to_csv("data/sentiment_results.csv", index=False)

print(data)
print("\nResults saved to data/sentiment_results.csv")
