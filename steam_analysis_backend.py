from flask import Flask, render_template, request, jsonify
from game_info_collector import GameTextData
import re
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import threading
from nlp import TextProcessor
from wordcloud import WordCloud
import numpy
from io import BytesIO
import base64
from matplotlib.figure import Figure
import numpy as np

app = Flask(__name__)

api_cache = {}
cache_timeout = 3000
text_processor = TextProcessor()


class GameData:

    game_info = ""
    review_info = ""
    review_data = ""


game_datas = GameData()


def clean_cache():

    while True:
        current_time = time.time()
        keys_cache = list(api_cache.keys())

        for key in keys_cache:
            if key in api_cache and current_time - api_cache[key]["timestamp"]:
                del api_cache[key]

        time.sleep(300)


@app.before_request
def start_cache_cleaning():
    cache_thread = threading.Thread(target=clean_cache, daemon=True)
    cache_thread.start()


def aggregate_by_period(df, period):
    df_agg = df.copy()
    df_agg["date_of_creation"] = pd.to_datetime(df_agg["date_of_creation"])

    if period == "yearly":
        df_agg["period"] = df_agg["date_of_creation"].dt.year
        title = "Yearly Sentiment Analysis"
    elif period == "monthly":
        df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%m")
        title = "Monthly Sentiment Analysis"
    elif period == "daily":
        # Filter to only the last 30 days for daily view
        df_agg = df_agg[
            df_agg["date_of_creation"]
            >= (df_agg["date_of_creation"].max() - timedelta(days=30))
        ].copy()
        df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%m-%d")
        title = "Daily Sentiment Analysis (Last 30 Days)"
    else:
        raise ValueError("Period must be one of: yearly, monthly, weekly, daily")

    # Group by period and review type
    grouped = (
        df_agg.groupby(["period", "review_rating"]).size().reset_index(name="count")
    )

    # Pivot the data to get review types as columns
    pivot_df = (
        grouped.pivot(index="period", columns="review_rating", values="count")
        .fillna(0)
        .reset_index()
    )

    # Ensure both Positive and Negative columns exist
    if "Positive" not in pivot_df.columns:
        pivot_df["Positive"] = 0
    if "Negative" not in pivot_df.columns:
        pivot_df["Negative"] = 0

    # Sort by period
    pivot_df = pivot_df.sort_values("period")

    return pivot_df


@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("steam_analysis_landing.html")


@app.route("/submit", methods=["POST", "GET"])
def submit():
    username = request.values.get("gameUrl")
    app_id = re.search(r"app\/(\d+)", username)[1]

    cache_key = f"app_{app_id}"
    if (
        cache_key in api_cache
        and time.time() - api_cache[cache_key]["timestamp"] < cache_timeout
    ):
        game_stats = api_cache[cache_key]["game_stats"]
        review_data = api_cache[cache_key]["review_data"]
        print("Using cached data for", app_id)
    else:
        # Not in cache, fetch from API
        print(f"Fetching data for app ID: {app_id}")

        game_data = GameTextData(app_id, language="english")
        game_stats, review_data = game_data.get_all_data()

        game_datas.game_info = game_stats
        game_datas.review_data = review_data

        api_cache[cache_key] = {
            "game_stats": game_stats,
            "review_data": review_data,
            "timestamp": time.time(),
        }
    # chart_data = create_sentiment_chart_data(review_data)
    pos_wc = create_wordcloud(review_data, True)
    neg_wc = create_wordcloud(review_data, False)

    return render_template(
        "steam_analysis_dashboard.html",
        game_info=game_stats,
        pos_wc_data=pos_wc,
        neg_wc_data=neg_wc,
    )


def create_wordcloud(reviews, color_pos_neg: bool):

    vote_scores = list(map(lambda review: review["weighted_vote_score"], reviews))
    sorted_voted_index = np.argsort(vote_scores)[::-1]

    sorted_reviews = [reviews[i] for i in sorted_voted_index]
    wc_review_data = list(
        filter(
            lambda user_review: user_review["voted_up"] is color_pos_neg,
            sorted_reviews[:400],
        )
    )

    text_processor = TextProcessor()
    processed_texts = text_processor.process_texts(
        list(map(lambda x: x["review"], wc_review_data))
    )
    colormap = "Greens" if color_pos_neg else "OrRd"

    word_dict = " ".join(numpy.concatenate(processed_texts).tolist())
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
    ).generate(word_dict)

    # Display the wordcloud
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")

    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/topic_modelling_table")
def get_important_topics():

    pos_review_data = list(
        filter(
            lambda user_review: user_review["voted_up"] is True,
            game_datas.review_data,
        )
    )
    pos_processed_texts = text_processor.process_texts(
        list(map(lambda x: x["review"], pos_review_data))
    )
    pos_topics_dict = text_processor.get_topics(pos_processed_texts, n_gram_words=5)

    pos_topic_df = pd.DataFrame(pos_topics_dict)

    pos_topic_html = pos_topic_df.to_html(
        classes="table table-striped table-hover", index=False
    )

    return jsonify({"html_table": pos_topic_html})


@app.get("/sentiment_chart_data")
def create_sentiment_chart_data():
    # Extract sentiment data from game stats
    game_stats = game_datas.review_data
    sentiment_dict = {
        "review_rating": [
            "Positive" if review["voted_up"] else "Negative" for review in game_stats
        ],
        "date_of_creation": [
            datetime.fromtimestamp(review["timestamp_created"]).strftime("%Y-%m-%d")
            for review in game_stats
        ],
    }

    sentiment_df = pd.DataFrame(sentiment_dict)

    time_filter = request.args.get("time_filter", "yearly")

    # Get aggregated data
    daily_df = aggregate_by_period(sentiment_df, time_filter)

    # Define colors for sentiment categories
    colors = {
        "Positive": "rgba(40, 167, 69, 0.7)",  # Green
        "Negative": "rgba(220, 53, 69, 0.7)",  # Red
    }

    # Create datasets for Chart.js
    datasets = []
    for category in ["Positive", "Negative"]:
        datasets.append(
            {
                "label": category,
                "data": daily_df[category].tolist(),
                "backgroundColor": colors[category],
            }
        )

    return {"labels": daily_df["period"].tolist(), "datasets": datasets}


if __name__ == "__main__":
    app.run(debug=True)
