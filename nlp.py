import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
from sklearn.decomposition import LatentDirichletAllocation
from nltk.probability import FreqDist
from nltk import bigrams, ngrams


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")


def create_wordcloud(word_dict, colormap):
    return WordCloud(
        background_color="white",
        max_words=100,
        width=800,
        height=400,
        colormap=colormap,
        contour_width=1,
        contour_color="steelblue",
        max_font_size=100,
        random_state=42,
    ).generate_from_frequencies(word_dict)


# review_sample = review_data[:10]

# imp_review_data = []

# for reviews in review_sample:
#     imp_review_data.append(
#         {
#             "review_text": reviews["review"],
#             "review_id": reviews["recommendationid"],
#             "language": reviews["language"],
#             "date_of_creation": reviews["timestamp_created"],
#             "date_of_updated": reviews["timestamp_updated"],
#             "review_sentiment": (
#                 "Positive" if reviews["voted_up"] is True else "Negative"
#             ),
#             "weighted_vote_score": reviews["weighted_vote_score"],
#             "purchased_on_steam": reviews["steam_purchase"],
#             "received_for_free": reviews["received_for_free"],
#             "written_during_early_access": reviews["written_during_early_access"],
#             "played_on_steam_deck": reviews["primarily_steam_deck"],
#         }
#     )


class TextProcessor:
    def __init__(self):
        """Initialize the text processor with necessary NLTK components."""
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):

        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.lower()

        words = text.split()

        if len(words) < 50:
            processed_words = [""]
        else:
            processed_words = []
            for word in words:
                if word not in self.stop_words:
                    lemmatized_word = self.lemmatizer.lemmatize(word)
                    processed_words.append(lemmatized_word)
        return processed_words

    def process_texts(self, texts):

        processed_texts = []
        processed_texts.extend(self.clean_text(text) for text in texts)
        return processed_texts

    def get_word_freq(self, processed_texts, n=10):

        all_words = []
        # for words in processed_texts:
        #     all_words.extend(words)

        # Count word frequencies
        words = nltk.tokenize.word_tokenize(" ".join(processed_texts))
        fdist = FreqDist(ngrams(words, 2)).most_common(10)
        # for wird in fdist:
        #     all_words.append({"_".join(wird[0]), wird[1]})
        all_words.extend(
            {"bigram": "_".join(wird[0]), "count": wird[1]} for wird in fdist
        )

        print(all_words)
        # Return top n words
        return pd.DataFrame(all_words).sort_values(by="count", ascending=True)

    def generate_word_cloud(self, processed_texts):

        all_words = []
        for words in processed_texts:
            all_words.extend(words)

        # Join words into a space-separated string
        text = " ".join(all_words)

        # Generate word cloud
        return WordCloud(width=800, height=400, background_color="white").generate(text)

    def extract_ngrams(self, processed_texts, n=2):

        all_ngrams = []
        for words in processed_texts:
            all_ngrams.extend(list(ngrams(words, n)))

        # Return top 10 ngrams
        return Counter(all_ngrams)

    def get_topics(self, ngram_data, n_topics=7, n_gram_words=3):

        pre_fixed_data = list(map(lambda x: " ".join(x), ngram_data))

        vec = CountVectorizer(ngram_range=(2, n_gram_words))
        vec_data = vec.fit_transform(pre_fixed_data)

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=25,
            learning_method="batch",
            n_jobs=1,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )
        lda.fit_transform(vec_data)

        topic_words = []

        topic_no = []
        top_n = n_gram_words
        topic_collection = []

        # get the topics and each word per topic
        for idx, topic in enumerate(lda.components_):

            topic_words.extend(
                vec.get_feature_names_out()[i]
                for i in topic.argsort()[: -top_n - 1 : -1]
            )

            topic_collection.append(topic_words[:])
            topic_no.append(f"Topic {idx+1}")
            topic_words.clear()

        return dict(zip(topic_no, topic_collection))


if __name__ == "__main__":

    with open("test.json", "r+") as file:
        game_data = json.load(file)

    texts = list(
        filter(
            lambda user_review: user_review["language"] == "english",
            game_data["review_data"],
        )
    )

    texts = list(map(lambda user_reviews: user_reviews["review"], texts))

    # Create a text processor instance
    processor = TextProcessor()

    # # Process the texts
    processed_texts = processor.process_texts(texts)
    for i, tokens in enumerate(processed_texts):
        print(f"Text {i+1}: {tokens}")

    # # Get word frequencies
    print("\nTop 10 words by frequency:")
    pro_text = list(map(lambda x: " ".join(x), processed_texts))
    word_freqs = processor.get_word_freq(pro_text)
    for i, (word, freq) in enumerate(dict(word_freqs).items()):
        if i > 10:
            break
        print(f"{word}: {freq}")

    # # Generate word cloud
    wordcloud = processor.generate_word_cloud(processed_texts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()

    topic_data = processor.get_topics(processed_texts)
    print(topic_data)
