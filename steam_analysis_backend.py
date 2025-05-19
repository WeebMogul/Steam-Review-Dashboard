# from flask import Flask, render_template, request, jsonify
# from game_info_collector import GameTextData
# import re
# import plotly_express as px
# from datetime import datetime, timedelta
# import pandas as pd

# app = Flask(__name__)


# def aggregate_by_period(df, period):

#     df_agg = df.copy()
#     df_agg["date_of_creation"] = pd.to_datetime(df_agg["date_of_creation"])

#     if period == "yearly":
#         df_agg["period"] = df_agg["date_of_creation"].dt.year
#         title = "Yearly Sentiment Analysis"
#     elif period == "monthly":
#         df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%m")
#         title = "Monthly Sentiment Analysis"
#     elif period == "weekly":
#         df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%U")
#         title = "Weekly Sentiment Analysis"
#     elif period == "daily":
#         # Filter to only the last 7 days for daily view
#         df_agg = df_agg[
#             df_agg["date_of_creation"]
#             >= (df_agg["date_of_creation"].max() - timedelta(days=30))
#         ].copy()
#         df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%m-%d")
#         # title = "Daily Sentiment Analysis (Last 7 Days)"
#     else:
#         raise ValueError("Period must be one of: yearly, monthly, weekly, daily")

#     # Group by period and review type
#     grouped = (
#         df_agg.groupby(["period", "review_rating"], as_index=False)
#         .size()
#         .sort_values(by="period")
#         .reset_index()
#     )

#     return grouped


# @app.route("/")
# def index():
#     return render_template("steam_analysis_landing.html")


# @app.route("/submit", methods=["POST", "GET"])
# def submit():

#     username = request.values.get("gameUrl")
#     app_id = re.search(r"app\/(\d+)", username).group(1)
#     print(app_id)
#     game_data = GameTextData(app_id, language="English", total_reviews=100)
#     game_stats, review_data = game_data.get_all_data()
#     print("It's working")
#     chart_json_data = send_sentiment_data(review_data)
#     print(f"Working {chart_json_data}")

#     return render_template(
#         "steam_analysis_dashboard.html",
#         game_info=game_stats,
#         chart_data=chart_json_data,
#     )


# def create_dataframe_for_chartjs(df):

#     labels = df["period"].tolist()

#     categories = [col for col in df.columns if col != "period"]

#     datasets = []

#     colors = [
#         "rgba(255, 99, 132, 0.7)",  # Red
#         "rgba(54, 162, 235, 0.7)",  # Blue
#     ]

#     for i, category in enumerate(categories):
#         # Get the color for this category (cycle through colors if needed)
#         color_index = i % len(colors)

#         datasets.append(
#             {
#                 "label": category,
#                 "data": df[category].tolist(),
#                 "backgroundColor": colors[color_index],
#             }
#         )

#     chart_data = {"labels": labels, "datasets": datasets}

#     return chart_data


# def send_sentiment_data(game_stats):

#     sentiment_dict = {}

#     # for user_reviews in review_data:

#     sentiment_dict["review_rating"] = list(
#         map(
#             lambda x: "Positive" if x["voted_up"] is True else "Negative",
#             game_stats,
#         )
#     )
#     sentiment_dict["date_of_creation"] = list(
#         map(
#             lambda x: datetime.fromtimestamp(x["timestamp_created"]).strftime(
#                 "%Y-%m-%d"
#             ),
#             game_stats,
#         )
#     )

#     sentiment_df = pd.DataFrame(sentiment_dict)

#     yearly_df = aggregate_by_period(sentiment_df, "yearly")
#     monthly_df = aggregate_by_period(sentiment_df, "monthly")
#     daily_df = aggregate_by_period(sentiment_df, "daily")

#     chart_data = create_dataframe_for_chartjs(daily_df)

#     return jsonify(chart_data)


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from game_info_collector import GameTextData
import re
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)


def aggregate_by_period(df, period):
    df_agg = df.copy()
    df_agg["date_of_creation"] = pd.to_datetime(df_agg["date_of_creation"])

    if period == "yearly":
        df_agg["period"] = df_agg["date_of_creation"].dt.year
        title = "Yearly Sentiment Analysis"
    elif period == "monthly":
        df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%m")
        title = "Monthly Sentiment Analysis"
    elif period == "weekly":
        df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%U")
        title = "Weekly Sentiment Analysis"
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


@app.route("/")
def index():
    return render_template("steam_analysis_landing.html")


@app.route("/submit", methods=["POST", "GET"])
def submit():
    username = request.values.get("gameUrl")
    app_id = re.search(r"app\/(\d+)", username).group(1)
    print(app_id)
    game_data = GameTextData(app_id, language="all", total_reviews=1000)
    game_stats, review_data = game_data.get_all_data()
    print("It's working")
    chart_data = create_sentiment_chart_data(review_data)
    print(f"Working {chart_data}")

    return render_template(
        "steam_analysis_dashboard.html",
        game_info=game_stats,
        chart_data=json.dumps(chart_data),
    )


def create_sentiment_chart_data(game_stats):
    # Extract sentiment data from game stats
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

    # Get aggregated data
    daily_df = aggregate_by_period(sentiment_df, "daily")

    # Create Chart.js data format
    labels = daily_df["period"].tolist()

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

    chart_data = {"labels": labels, "datasets": datasets}

    return chart_data


if __name__ == "__main__":
    app.run(debug=True)
