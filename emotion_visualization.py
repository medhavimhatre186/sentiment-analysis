import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment results
data = pd.read_csv("data/sentiment_results.csv")

# Count emotions
emotion_counts = data['Emotion'].value_counts()

print(emotion_counts)

# Create emotion-wise bar chart
plt.figure()
emotion_counts.plot(kind='bar')

plt.title("Emotion-wise Analysis")
plt.xlabel("Emotion")
plt.ylabel("Number of Reviews")

plt.show()
