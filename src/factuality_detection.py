"""
Factuality Detection Module
Assesses the factuality and reliability of tweets
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, regexp_replace, lower, trim,
    sum as spark_sum, avg, count, lit
)
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType
import pandas as pd
import math
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import FACT_CHECK_KEYWORDS, FACTUALITY_THRESHOLDS


class FactualityDetector:
    """Detects factuality and reliability of Twitter content"""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize factuality detector
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        self.fact_check_keywords = FACT_CHECK_KEYWORDS
    
    def _calculate_keyword_score(self, text: str) -> float:
        """
        Calculate score based on fact-checking keywords
        
        Args:
            text: Input text
            
        Returns:
            Keyword score (0-1)
        """
        if not text:
            return 0.5  # Neutral score for empty text
        
        text_lower = text.lower()
        
        # Positive indicators (increase reliability)
        positive_keywords = [
            "verified", "fact-check", "factual", "reliable source",
            "confirmed", "official", "authentic", "credible"
        ]
        
        # Negative indicators (decrease reliability)
        negative_keywords = [
            "fake news", "misinformation", "disinformation",
            "unverified", "unconfirmed", "rumor", "alleged",
            "unsubstantiated", "hoax"
        ]
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        # Calculate score: -1 to 1, normalized to 0-1
        score = (positive_count - negative_count) / max(len(text.split()), 1)
        normalized_score = (score + 1) / 2  # Normalize to 0-1
        
        return min(max(normalized_score, 0.0), 1.0)
    
    def _calculate_user_credibility_score(self, followers: int, engagement: int) -> float:
        """
        Calculate user credibility based on followers and engagement
        
        Args:
            followers: Number of followers
            engagement: Total engagement (retweets + favorites)
            
        Returns:
            Credibility score (0-1)
        """
        # Normalize followers (log scale for large differences)
        if followers > 0:
            follower_score = min(math.log10(followers + 1) / 7, 1.0)  # Cap at 10M followers
        else:
            follower_score = 0.1
        
        # Engagement ratio
        if followers > 0:
            engagement_ratio = min(engagement / max(followers, 1), 0.1)  # Normalize
            engagement_score = min(engagement_ratio * 10, 1.0)
        else:
            engagement_score = 0.1
        
        # Combined credibility (weighted)
        credibility = (follower_score * 0.6) + (engagement_score * 0.4)
        
        return min(max(credibility, 0.0), 1.0)
    
    def _calculate_engagement_quality(self, retweets: int, favorites: int, replies: int) -> float:
        """
        Assess engagement quality
        
        Args:
            retweets: Number of retweets
            favorites: Number of favorites
            replies: Number of replies
            
        Returns:
            Quality score (0-1)
        """
        total_engagement = retweets + favorites + replies
        
        if total_engagement == 0:
            return 0.5  # Neutral for no engagement
        
        # High favorites-to-replies ratio might indicate quality content
        if replies > 0:
            favorite_reply_ratio = favorites / replies
        else:
            favorite_reply_ratio = favorites
        
        # Normalize
        normalized_ratio = min(favorite_reply_ratio / 10, 1.0)
        
        # Engagement volume factor
        volume_factor = min(math.log10(total_engagement + 1) / 5, 1.0)
        
        return (normalized_ratio * 0.5) + (volume_factor * 0.5)
    
    def _calculate_factuality_score(
        self, text: str, followers: int, retweets: int,
        favorites: int, replies: int
    ) -> tuple:
        """
        Calculate overall factuality score
        
        Args:
            text: Tweet text
            followers: Follower count
            retweets: Retweet count
            favorites: Favorite count
            replies: Reply count
            
        Returns:
            Tuple of (factuality_score, reliability_label, keyword_score, credibility_score, engagement_score)
        """
        keyword_score = self._calculate_keyword_score(text)
        engagement = retweets + favorites + replies
        credibility_score = self._calculate_user_credibility_score(followers, engagement)
        engagement_score = self._calculate_engagement_quality(retweets, favorites, replies)
        
        # Weighted combination
        factuality_score = (
            keyword_score * 0.4 +
            credibility_score * 0.3 +
            engagement_score * 0.3
        )
        
        # Determine reliability label
        if factuality_score >= FACTUALITY_THRESHOLDS["high_reliability"]:
            reliability_label = "high"
        elif factuality_score >= FACTUALITY_THRESHOLDS["medium_reliability"]:
            reliability_label = "medium"
        else:
            reliability_label = "low"
        
        return (
            float(factuality_score),
            reliability_label,
            float(keyword_score),
            float(credibility_score),
            float(engagement_score)
        )
    
    def detect_factuality(self, df):
        """
        Detect factuality for all tweets in DataFrame
        
        Args:
            df: Spark DataFrame with tweet data
            
        Returns:
            DataFrame with factuality columns added
        """
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import StructType as SparkStructType, StructField
        
        factuality_schema = SparkStructType([
            StructField("factuality_score", DoubleType(), True),
            StructField("reliability_label", StringType(), True),
            StructField("keyword_score", DoubleType(), True),
            StructField("credibility_score", DoubleType(), True),
            StructField("engagement_score", DoubleType(), True)
        ])
        
        @pandas_udf(factuality_schema)
        def detect_factuality_batch(batch: pd.DataFrame) -> pd.DataFrame:
            results = []
            for _, row in batch.iterrows():
                score, label, kw_score, cred_score, eng_score = self._calculate_factuality_score(
                    str(row.get('tweet_text', '')),
                    int(row.get('followers_count', 0)),
                    int(row.get('retweet_count', 0)),
                    int(row.get('favorite_count', 0)),
                    int(row.get('reply_count', 0))
                )
                results.append({
                    "factuality_score": score,
                    "reliability_label": label,
                    "keyword_score": kw_score,
                    "credibility_score": cred_score,
                    "engagement_score": eng_score
                })
            return pd.DataFrame(results)
        
        # Apply factuality detection
        factuality_df = df.withColumn(
            "factuality",
            detect_factuality_batch(
                F.struct(
                    col("tweet_text"),
                    col("followers_count"),
                    col("retweet_count"),
                    col("favorite_count"),
                    col("reply_count")
                )
            )
        )
        
        # Extract individual columns
        result_df = factuality_df.select(
            "*",
            col("factuality.factuality_score").alias("factuality_score"),
            col("factuality.reliability_label").alias("reliability_label"),
            col("factuality.keyword_score").alias("keyword_score"),
            col("factuality.credibility_score").alias("credibility_score"),
            col("factuality.engagement_score").alias("engagement_score")
        ).drop("factuality")
        
        return result_df
