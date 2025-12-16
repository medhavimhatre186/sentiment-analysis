import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment results
data = pd.read_csv("data/sentiment_results.csv")

# Count sentiment values
sentiment_counts = data['Sentiment'].value_counts()

print(sentiment_counts)

# Create bar graph
plt.figure()
sentiment_counts.plot(kind='bar')

plt.title("Sentiment Analysis Result")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Reviews")

plt.show()
