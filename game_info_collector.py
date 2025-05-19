import requests
import json
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, quote

s = requests.Session()


class GameTextData:

    def __init__(self, app_id, language="all", from_days=0, total_reviews=1000):
        self.app_id = app_id
        self.steam_game_url = (
            f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=ae"
        )
        self.base_review_url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&num_per_page=100"
        self.language = language
        self.from_days = from_days
        self.game_info = {}
        self.review_info = {}
        self.till_date = from_days
        self.total_reviews = total_reviews

    def _get_game_info(self):

        game_json = json.loads(s.get(self.steam_game_url).text)
        game_data = game_json[f"{self.app_id}"]["data"]

        self.game_info["game_name"] = game_data["name"]
        self.game_info["required_age"] = game_data["required_age"]
        self.game_info["game_publishers"] = game_data["publishers"]
        self.game_info["game_developers"] = game_data["developers"]

        if game_data["is_free"] == False:
            self.game_info["game_price"] = game_data["price_overview"][
                "final_formatted"
            ]
        else:
            self.game_info["game_price"] = "Free"

        self.game_info["game_categories"] = list(
            map(lambda category: category["description"], game_data["categories"])
        )

        self.game_info["game_genres"] = list(
            map(lambda category: category["description"], game_data["genres"])
        )

        self.game_info["header_image"] = game_data["header_image"]
        self.game_info["short_description"] = game_data["short_description"]

    def _get_review_info(self):

        review_data = json.loads(s.get(url=self.base_review_url).text)

        self.review_info["positive_review_count"] = review_data["query_summary"][
            "total_positive"
        ]
        self.review_info["negative_review_count"] = review_data["query_summary"][
            "total_negative"
        ]
        self.review_info["total_review_count"] = review_data["query_summary"][
            "total_reviews"
        ]

        self.review_info["perc_positive"] = round(
            (
                self.review_info["positive_review_count"]
                / self.review_info["total_review_count"]
            )
            * 100,
            2,
        )
        self.review_info["perc_negative"] = round(
            (
                self.review_info["negative_review_count"]
                / self.review_info["total_review_count"]
            )
            * 100,
            2,
        )

    # def game_info(self):

    #     return self._get_game_info() | self._get_review_info()

    def _get_review_data(self):

        cursor = "*"
        review_texts = []

        for i in range(0, self.review_info["total_review_count"]):

            steam_url = (
                self.base_review_url
                + f"&language={self.language}&day_range={self.till_date}&cursor={cursor}"
            )

            review_json = json.loads(s.get(url=steam_url).text)
            review_texts.extend(review_json["reviews"])

            cursor = quote(review_json["cursor"])

            if len(review_texts) > self.total_reviews:
                break

        return review_texts

    def get_all_data(self):

        self._get_game_info()
        self._get_review_info()

        full_data = self.game_info | self.review_info
        full_data["review_data"] = self._get_review_data()

        return full_data

    # steam_url = base_steam_url + f"&cursor={quote(cursor)}"
