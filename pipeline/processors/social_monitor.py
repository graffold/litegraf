"""Social media monitoring for biomarker mentions on Twitter/X."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class TweetMetadata:
    """Metadata for a tweet mentioning biomarkers."""

    tweet_id: str
    text: str
    author_username: str
    author_id: str
    created_at: str
    retweet_count: int
    like_count: int
    reply_count: int
    url: str
    mentioned_proteins: list[str] | None = None
    mentioned_diseases: list[str] | None = None


class SocialMonitor:
    """Monitor Twitter/X for biomarker mentions using Twitter API v2.

    Requires Twitter API v2 credentials (bearer token or OAuth 2.0).
    Rate limits: 450 requests per 15 minutes for search endpoint.
    """

    BASE_URL = "https://api.twitter.com/2"

    def __init__(
        self,
        bearer_token: str,
        rate_limit_delay: float = 2.0,
        max_retries: int = 3,
    ):
        """Initialize the social monitor.

        Args:
            bearer_token: Twitter API v2 bearer token
            rate_limit_delay: Delay in seconds between API requests (default 2.0s = 30 req/min)
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.bearer_token = bearer_token
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time = 0.0

    async def search_tweets(
        self,
        keywords: list[str],
        max_results: int = 100,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[TweetMetadata]:
        """Search tweets by keywords.

        Args:
            keywords: List of keywords to search for (combined with OR)
            max_results: Maximum number of tweets to return (10-100 per request)
            start_time: Start time in ISO 8601 format (e.g., "2024-01-01T00:00:00Z")
            end_time: End time in ISO 8601 format

        Returns:
            List of TweetMetadata objects
        """
        query = " OR ".join(keywords)
        params: dict[str, Any] = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username",
        }

        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        return await self._fetch_tweets(params)

    async def monitor_keywords(
        self,
        keywords: list[str],
        interval_seconds: int = 300,
        max_iterations: int | None = None,
    ) -> list[TweetMetadata]:
        """Monitor keywords continuously at specified interval.

        Args:
            keywords: List of keywords to monitor
            interval_seconds: Polling interval in seconds (default 5 minutes)
            max_iterations: Maximum number of polling iterations (None = infinite)

        Returns:
            List of all collected TweetMetadata objects
        """
        all_tweets: list[TweetMetadata] = []
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            logger.info(f"Monitoring iteration {iteration + 1}, keywords: {keywords}")
            tweets = await self.search_tweets(keywords, max_results=100)
            all_tweets.extend(tweets)
            logger.info(f"Found {len(tweets)} new tweets")

            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                await asyncio.sleep(interval_seconds)

        return all_tweets

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _fetch_tweets(self, params: dict[str, Any]) -> list[TweetMetadata]:
        """Fetch tweets from Twitter API with retry logic."""
        await self._enforce_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

        url = f"{self.BASE_URL}/tweets/search/recent"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limit exceeded, retrying...")
                    raise aiohttp.ClientError("Rate limit exceeded")

                if response.status != 200:
                    logger.error(f"Twitter API error: {response.status}")
                    return []

                data = await response.json()
                return self._parse_tweets(data)

    def _parse_tweets(self, data: dict[str, Any]) -> list[TweetMetadata]:
        """Parse Twitter API response into TweetMetadata objects."""
        tweets: list[TweetMetadata] = []

        if "data" not in data:
            return tweets

        # Build user lookup dict
        users = {}
        if "includes" in data and "users" in data["includes"]:
            for user in data["includes"]["users"]:
                users[user["id"]] = user["username"]

        for tweet_data in data["data"]:
            tweet_id = tweet_data["id"]
            text = tweet_data["text"]
            author_id = tweet_data["author_id"]
            author_username = users.get(author_id, "unknown")
            created_at = tweet_data["created_at"]

            metrics = tweet_data.get("public_metrics", {})
            retweet_count = metrics.get("retweet_count", 0)
            like_count = metrics.get("like_count", 0)
            reply_count = metrics.get("reply_count", 0)

            url = f"https://twitter.com/{author_username}/status/{tweet_id}"

            tweet = TweetMetadata(
                tweet_id=tweet_id,
                text=text,
                author_username=author_username,
                author_id=author_id,
                created_at=created_at,
                retweet_count=retweet_count,
                like_count=like_count,
                reply_count=reply_count,
                url=url,
            )
            tweets.append(tweet)

        return tweets

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()
