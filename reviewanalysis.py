import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer


data = {
    'review': [
        "I absolutely love this phone! The camera is amazing.",
        "Terrible battery life. I regret buying this.",
        "It's okay. Good value for money but the screen is dim.",
        "Worst customer service ever. Very rude.",
        "Fantastic quality and fast shipping. Highly recommend!",
        "The product arrived broken. Very disappointed.",
        "Great features, but a bit expensive.",
        "I like the design, but the performance is slow.",
        "Best purchase I made this year. Five stars!",
        "Not worth the price. Do not buy."
    ]
}

df = pd.DataFrame(data)

print("--- Raw Reviews ---")
print(df.head())
print("\n")


def get_sentiment(text):
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'


df['sentiment'] = df['review'].apply(get_sentiment)

print("--- Analyzed Data ---")
print(df)
print("\n")


print("--- Extracting Top Keywords... ---")


vectorizer = CountVectorizer(stop_words='english')


matrix = vectorizer.fit_transform(df['review'])


counts = pd.DataFrame(
    matrix.toarray(), 
    columns=vectorizer.get_feature_names_out()
).sum().sort_values(ascending=False)

print("Top 5 Keywords mentioned by customers:")
print(counts.head(5))


print("\nGenerating Chart... (Check popup)")


sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
colors = ['green', 'red', 'gray'] 

sentiment_counts.plot(kind='bar', color=colors, rot=0)

plt.title('Customer Sentiment Analysis', fontsize=14)
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.grid(axis='y', linestyle='--', alpha=0.7)


for index, value in enumerate(sentiment_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontweight='bold')

plt.show()