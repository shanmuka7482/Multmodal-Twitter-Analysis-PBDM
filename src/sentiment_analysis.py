"""
Sentiment Analysis Module
Uses VADER and TextBlob for sentiment detection
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import StringType, DoubleType
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd


class SentimentAnalyzer:
    """Performs sentiment analysis on Twitter data"""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize sentiment analyzer
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        self.vader = SentimentIntensityAnalyzer()
        
        # Download required NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
    
    def _get_vader_sentiment(self, text: str) -> dict:
        """
        Get VADER sentiment scores
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or text.strip() == "":
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
        
        scores = self.vader.polarity_scores(text)
        return scores
    
    def _get_textblob_sentiment(self, text: str) -> float:
        """
        Get TextBlob sentiment polarity
        
        Args:
            text: Input text
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        if not text or text.strip() == "":
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0
    
    def _analyze_sentiment_combined(self, text: str) -> tuple:
        """
        Combined sentiment analysis using VADER and TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, compound_score, polarity_score, confidence)
        """
        if not text or text.strip() == "":
            return ("neutral", 0.0, 0.0, 0.0)
        
        # Get VADER scores
        vader_scores = self._get_vader_sentiment(text)
        compound = vader_scores["compound"]
        
        # Get TextBlob polarity
        polarity = self._get_textblob_sentiment(text)
        
        # Combine scores (weighted average)
        combined_score = (compound * 0.6) + (polarity * 0.4)
        
        # Determine label
        if combined_score >= 0.05:
            label = "positive"
            confidence = abs(combined_score)
        elif combined_score <= -0.05:
            label = "negative"
            confidence = abs(combined_score)
        else:
            label = "neutral"
            confidence = 1 - abs(combined_score)
        
        return (label, float(compound), float(polarity), float(confidence))
    
    def analyze_sentiment(self, df):
        """
        Analyze sentiment for all tweets in DataFrame
        
        Args:
            df: Spark DataFrame with 'tweet_text' column
            
        Returns:
            DataFrame with sentiment columns added
        """
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import StructType as SparkStructType, StructField
        
        sentiment_schema = SparkStructType([
            StructField("sentiment_label", StringType(), True),
            StructField("sentiment_compound", DoubleType(), True),
            StructField("sentiment_polarity", DoubleType(), True),
            StructField("sentiment_confidence", DoubleType(), True)
        ])
        
        @pandas_udf(sentiment_schema)
        def analyze_sentiment_batch(texts: pd.Series) -> pd.DataFrame:
            results = []
            for text in texts:
                label, compound, polarity, confidence = self._analyze_sentiment_combined(str(text))
                results.append({
                    "sentiment_label": label,
                    "sentiment_compound": compound,
                    "sentiment_polarity": polarity,
                    "sentiment_confidence": confidence
                })
            return pd.DataFrame(results)
        
        # Apply the pandas UDF
        sentiment_df = df.withColumn(
            "sentiment",
            analyze_sentiment_batch(col("tweet_text"))
        )
        
        # Extract individual columns
        result_df = sentiment_df.select(
            "*",
            col("sentiment.sentiment_label").alias("sentiment_label"),
            col("sentiment.sentiment_compound").alias("sentiment_compound"),
            col("sentiment.sentiment_polarity").alias("sentiment_polarity"),
            col("sentiment.sentiment_confidence").alias("sentiment_confidence")
        ).drop("sentiment")
        
        return result_df
