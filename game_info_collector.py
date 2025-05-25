import requests
import json
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, quote
from concurrent.futures import ThreadPoolExecutor


class GameTextData:

    def __init__(
        self,
        app_id,
        total_reviews=None,
        language="all",
        from_days=9223372036854775807,
    ):
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
        self.session = requests.Session()
        self.headers = {
            "Cookie": "browserid=332813660915616257; steamCountry=AE%7C3619e5ce7302a0532b2a87b7028eee57"
        }

    def _get_game_info(self):

        game_json = json.loads(
            self.session.get(self.steam_game_url, headers=self.headers).text
        )
        game_data = game_json[f"{self.app_id}"]["data"]

        self.game_info["game_name"] = game_data["name"]
        self.game_info["required_age"] = game_data["required_age"]
        self.game_info["game_publishers"] = game_data["publishers"]
        self.game_info["game_developers"] = game_data["developers"]
        self.game_info["game_market_url"] = (
            f"https://store.steampowered.com/app/{self.app_id}"
        )

        if game_data["is_free"] is False:
            self.game_info["game_price"] = game_data["price_overview"][
                "final_formatted"
            ]
        else:
            self.game_info["game_price"] = "Free"

        if game_data["release_date"]["coming_soon"] is False:
            self.game_info["game_release_date"] = game_data["release_date"]["date"]
        else:
            self.game_info["game_release_date"] = "Coming Soon"

        self.game_info["game_categories"] = list(
            map(lambda category: category["description"], game_data["categories"])
        )

        self.game_info["game_genres"] = list(
            map(lambda category: category["description"], game_data["genres"])
        )

        self.game_info["header_image"] = game_data["header_image"]
        self.game_info["short_description"] = game_data["short_description"]

        return self.game_info

    def _get_review_info(self):

        review_data = json.loads(self.session.get(url=self.base_review_url).text)

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

        return self.review_info

    # def game_info(self):

    #     return self._get_game_info() | self._get_review_info()

    def _get_review_data(self):

        cursor = "*"
        review_texts = []

        if self.total_reviews is None:
            self.total_reviews = self.review_info["total_review_count"]

        num_requests = max(5, ((self.total_reviews + 99) // 100))

        for _ in range(num_requests):

            steam_url = f"{self.base_review_url}&language={self.language}&day_range={self.till_date}&cursor={cursor}"

            review_json = json.loads(self.session.get(url=steam_url).text)
            new_reviews = review_json.get("reviews", [])
            review_texts.extend(new_reviews)

            cursor = quote(review_json["cursor"])

            if len(review_texts) >= self.total_reviews:
                break

        return review_texts[: self.total_reviews]

    def get_all_data(self):

        with ThreadPoolExecutor(max_workers=1) as executor:
            game_info_exec = executor.submit(self._get_game_info)
            game_info_data = game_info_exec.result()
            review_info_exec = executor.submit(self._get_review_info)
            review_info_data = review_info_exec.result()

        full_data = {**game_info_data, **review_info_data}
        review_data = self._get_review_data()

        if not review_data:
            full_data["debug_message"] = (
                "No reviews were retrieved. The game may be new or have privacy restrictions."
            )
        return full_data, review_data

    # steam_url = base_steam_url + f"&cursor={quote(cursor)}"
