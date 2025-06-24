import streamlit as st
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nlp import TextProcessor
import numpy
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import re
from game_info_collector import GameTextData
import calendar

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")


def create_wordcloud(
    word_dict,
    colormap,
):

    word_dict = " ".join(numpy.concatenate(word_dict).tolist())
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
    ).generate(word_dict)


def render_word_count(sentiment_df):

    print(sentiment_df)


@st.fragment
def render_chart(sentiment_df):

    yearly_df = aggregate_by_period(sentiment_df, "yearly")
    monthly_df = aggregate_by_period(sentiment_df, "monthly")
    daily_df = aggregate_by_period(sentiment_df, "daily")

    time_period = st.radio("Select Time Period", ["Yearly", "Monthly", "Daily"])

    # Update the figure based on the selection
    if time_period == "Yearly":
        df = yearly_df
        x_title = "Year"
        y_title = "Count"
        dtick = 1
        fig = px.bar(df, x="period", y="size", color="review_rating")

    elif time_period == "Monthly":
        df = monthly_df
        x_title = "Month"
        y_title = "Count"
        dtick = None
        fig = px.bar(df, x="period", y="size", color="review_rating")
        fig.update_xaxes(tickmode="array", tickvals=df["period"], tickformat="%b %Y")

    else:  # Daily
        df = daily_df
        x_title = "Day"
        y_title = "Count"
        dtick = None
        dates = df["period"].tolist()
        tick_indices = list(range(0, len(dates), 1))
        tick_values = [dates[i] for i in tick_indices if i < len(dates)]

        fig = px.bar(df, x="period", y="size", color="review_rating")
        fig.update_xaxes(tickmode="array", tickvals=tick_values, tickformat="%d %b")

    # Create the figure with the selected data

    # Update layout (without the updatemenus)
    layout_updates = {"xaxis_title": x_title, "yaxis_title": y_title}
    if dtick is not None:
        layout_updates["xaxis_dtick"] = dtick
    fig.update_layout(**layout_updates)

    st.plotly_chart(fig)

    return fig


@st.fragment
def bar_and_cloud(review_data, sentiment):

    text_processor = TextProcessor()

    review_data = list(
        filter(
            lambda user_review: (
                user_review["voted_up"] is True
                if sentiment == "Positive"
                else user_review["voted_up"] is False
            ),
            review_data,
        )
    )

    processed_texts = text_processor.process_texts(
        list(map(lambda x: x["review"], review_data))
    )

    pro_text = list(map(lambda x: " ".join(x), processed_texts))

    col1, col2 = st.columns(2)
    with col1:
        word_freqs = text_processor.get_word_freq(pro_text)

        fig = px.bar(word_freqs, y="bigram", x="count", orientation="h")

        key_plot = "negative" if sentiment == "Negative" else "positive"

        st.plotly_chart(fig, key=key_plot, use_container_width=True)

    with col2:

        negative_wc = create_wordcloud(
            processed_texts, "OrRd" if sentiment == "Negative" else "Greens"
        )

        # Display the wordcloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(negative_wc, interpolation="bilinear")
        ax.axis("off")
        st.markdown("")
        st.pyplot(fig)
        plt.close()


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
        # Filter to only the last 7 days for daily view
        df_agg = df_agg[
            df_agg["date_of_creation"]
            >= (df_agg["date_of_creation"].max() - timedelta(days=30))
        ].copy()
        df_agg["period"] = df_agg["date_of_creation"].dt.strftime("%Y-%m-%d")
        # title = "Daily Sentiment Analysis (Last 7 Days)"
    else:
        raise ValueError("Period must be one of: yearly, monthly, weekly, daily")

    # Group by period and review type
    return (
        df_agg.groupby(["period", "review_rating"], as_index=False)
        .size()
        .sort_values(by="period")
        .reset_index()
    )


st.markdown(
    """
<style>
    .stButton button {
        height: 2.75rem;  /* Match text input height */
        margin-top: 0px;  /* Remove top margin */
        width: 100%;      /* Make button use full width of column */
    }
    /* Remove padding above button */
    div[data-testid="column"] div.stButton {
        padding-top: 0px;
    }

    .stMainBlockContainer {
            max-width:50rem;
    }
    
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin: 10px 0px;
        border : 5px black solid;
        display:flex;
        flex-direction:column;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-align:center;
    }
    .card-text {
        color: #555;
        margin-bottom: 15px;
    }
    .card-button {
        background-color: #4CAF50;
        color: white;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 4px;
        padding: 10px 15px;
        cursor: pointer;
        border: none;
    }
    
    img{
        width:100%
    }

""",
    unsafe_allow_html=True,
)

st.title("Steam Review Analysis")

url = st.text_input("Enter URL", placeholder="Enter the URL")

st.sidebar.title("Filter ")
no_of_comments = st.sidebar.radio(
    "Select amount of comments", [1000, 5000, 10000, "all_comments"]
)
date_input = st.sidebar.date_input(
    "Enter Date", min_value=(datetime.now() - timedelta(365)), max_value=datetime.now()
)

no_of_days = datetime.now().day - date_input.day

if st.button("Click"):

    app_id = re.search(r"app\/(\d+)", url)[1]
    st.toast("Warming up...")
    game_data = GameTextData(
        app_id, language="english", total_reviews=no_of_comments, from_days=no_of_days
    )
    game_json, review_data = game_data.get_all_data()

    # Create a card using markdown with HTML
    st.markdown(
        f"""
    <div class="card">
        <div class="card-banner">
        <img src={game_json["header_image"]}>
        </div>
        <div class="card-content">
        <div class="card-title">{game_json["game_name"]}</div>
        <div class="card-text">
            {game_json["short_description"]}
            <br>
            Genres: {','.join(game_json["game_genres"])}
            <br>
            Price: {game_json["game_price"]}
        </div>
        <!--- <button class="card-button">Learn More</button> --!>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Split for positive and negative speeches

    pos_tab, neg_tab = st.tabs(["Positive", "Negative"])

    with pos_tab:

        bar_and_cloud(review_data, "Positive")

    with neg_tab:

        bar_and_cloud(review_data, "Negative")

    # st.markdown(
    #     "<div class='card'><div class='card-title'>Sentiment Over Time</div>",
    #     unsafe_allow_html=True,
    # )
    sentiment_dict = {
        "review_rating": list(
            map(
                lambda x: "Positive" if x["voted_up"] is True else "Negative",
                review_data,
            )
        ),
        "date_of_creation": list(
            map(
                lambda x: datetime.fromtimestamp(x["timestamp_created"]).strftime(
                    "%Y-%m-%d"
                ),
                review_data,
            )
        ),
    }

    sentiment_df = pd.DataFrame(sentiment_dict)
    render_chart(sentiment_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown(
    #     "<div class='card'><div class='card-title'>Most Relevant Topics</div>",
    #     unsafe_allow_html=True,
    # )

    text_processor = TextProcessor()
    positive_tab, negative_tab = st.tabs(["Positive Topics", "Negative Topics"])

    with positive_tab:

        text_processor = TextProcessor()

        review_data = list(
            filter(
                lambda user_review: (user_review["voted_up"] is True),
                review_data,
            )
        )

        processed_texts = text_processor.process_texts(
            list(map(lambda x: x["review"], review_data))
        )

        pro_text = list(map(lambda x: " ".join(x), processed_texts))
        st.table(text_processor.get_topics(processed_texts, n_gram_words=5))

    with negative_tab:
        text_processor = TextProcessor()

        review_data = list(
            filter(
                lambda user_review: (user_review["voted_up"] is False),
                review_data,
            )
        )

        processed_texts = text_processor.process_texts(
            list(map(lambda x: x["review"], review_data))
        )

        pro_text = list(map(lambda x: " ".join(x), processed_texts))
        st.table(text_processor.get_topics(processed_texts, n_gram_words=5))

    # st.markdown("</div>", unsafe_allow_html=True)
