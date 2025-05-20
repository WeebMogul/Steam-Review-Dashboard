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

steamReviewStopwords = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "game",
    "play",
    "played",
    "playing",
    "gameplay",
    "games",
    "steam",
    "review",
    "reviews",
    "player",
    "players",
    "hours",
    "hour",
    "playtime",
    "playthrough",
    "playthroughs",
    "session",
    "sessions",
    "buy",
    "bought",
    "purchase",
    "purchased",
    "purchasing",
    "price",
    "paid",
    "recommend",
    "recommended",
    "recommendation",
    "worth",
    "worthwhile",
    "experience",
    "experiences",
    "experienced",
    "time",
    "times",
    "minute",
    "minutes",
    "second",
    "seconds",
    "dlc",
    "content",
    "update",
    "updates",
    "patch",
    "patches",
    "version",
    "developer",
    "developers",
    "dev",
    "devs",
    "publisher",
    "publishers",
    "run",
    "runs",
    "running",
    "ran",
    "feels",
    "feeling",
    "feel",
    "felt",
    "like",
    "liked",
    "likes",
    "liking",
    "love",
    "loves",
    "loved",
    "loving",
    "hate",
    "hates",
    "hated",
    "hating",
    "good",
    "bad",
    "great",
    "terrible",
    "awesome",
    "amazing",
    "awful",
    "horrible",
    "better",
    "best",
    "worse",
    "worst",
    "just",
    "really",
    "very",
    "quite",
    "pretty",
    "extremely",
    "somewhat",
    "overall",
    "basically",
    "think",
    "thought",
    "thinking",
    "thinks",
    "get",
    "gets",
    "getting",
    "got",
    "gotten",
    "use",
    "uses",
    "using",
    "used",
    "need",
    "needs",
    "needed",
    "needing",
    "want",
    "wants",
    "wanted",
    "wanting",
    "try",
    "tries",
    "tried",
    "trying",
    "way",
    "ways",
    "one",
    "ones",
    "two",
    "three",
    "thing",
    "things",
    "make",
    "makes",
    "making",
    "made",
    "see",
    "sees",
    "seeing",
    "saw",
    "seen",
    "look",
    "looks",
    "looking",
    "looked",
    "well",
    "fine",
    "even",
    "still",
    "though",
    "although",
    "however",
    "also",
    "too",
    "first",
    "last",
    "next",
    "previous",
    "game",
]


def create_wordcloud(word_dict, colormap):
    wc = WordCloud(
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

    return wc


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
        text = re.sub(r"[^\w\s\']", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.lower()

        words = text.split()

        if len(words) >= 50:

            processed_words = []
            for word in words:
                if word not in self.stop_words and word not in steamReviewStopwords:
                    lemmatized_word = self.lemmatizer.lemmatize(word)
                    processed_words.append(lemmatized_word)

            return processed_words
        else:
            return [" "]

    def process_texts(self, texts):

        processed_texts = []
        for text in texts:
            processed_texts.append(self.clean_text(text))
        return processed_texts

    def get_word_freq(self, processed_texts, n=10):

        all_words = []
        for words in processed_texts:
            all_words.extend(words)

        # Count word frequencies
        word_freq = Counter(all_words)

        # Return top n words
        return word_freq.most_common(n)

    def generate_word_cloud(self, processed_texts):

        all_words = []
        for words in processed_texts:
            all_words.extend(words)

        # Join words into a space-separated string
        text = " ".join(all_words)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        return wordcloud

    def extract_ngrams(self, processed_texts, n=2):

        all_ngrams = []
        for words in processed_texts:
            all_ngrams.extend(list(ngrams(words, n)))

        # Count ngram frequencies
        ngram_freq = Counter(all_ngrams)

        # Return top 10 ngrams
        return ngram_freq

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

            for i in topic.argsort()[: -top_n - 1 : -1]:
                topic_words.append((vec.get_feature_names_out()[i]))

            topic_collection.append(topic_words[:])
            topic_no.append(f"Topic {idx+1}")
            topic_words.clear()

        return dict(zip(topic_no, topic_collection))


if __name__ == "__main__":

    with open("test.json", "r+") as file:
        game_data = json.load(file)

    texts = list(
        map(lambda user_review: user_review["review"], game_data["review_data"])
    )

    # Create a text processor instance
    processor = TextProcessor()

    # # Process the texts
    processed_texts = processor.process_texts(texts)
    print("Processed texts:")
    for i, tokens in enumerate(processed_texts):
        print(f"Text {i+1}: {tokens}")

    # # Get word frequencies
    print("\nTop 10 words by frequency:")
    word_freqs = processor.get_word_freq(processed_texts)
    for word, freq in word_freqs:
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
