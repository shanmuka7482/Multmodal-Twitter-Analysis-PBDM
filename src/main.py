"""
Main Execution Script for Twitter Multimodal Analysis
"""
import argparse
import sys
import os
import platform
from pathlib import Path

# Set Java options BEFORE importing PySpark (critical for Java 21+ compatibility)
if platform.system() == "Windows":
    java_opts = [
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
        "-Dio.netty.tryReflectionSetAccessible=true"
    ]
    # Only set if not already set
    if "_JAVA_OPTIONS" not in os.environ:
        os.environ["_JAVA_OPTIONS"] = " ".join(java_opts)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import SPARK_CONFIG, PROJECT_ROOT
from src.data_ingestion import TwitterDataIngestion
from src.sentiment_analysis import SentimentAnalyzer
from src.factuality_detection import FactualityDetector
from src.visualization import Visualizer


def create_spark_session(mode: str = "local", partitions: int = 4):
    """
    Create and configure Spark session
    
    Args:
        mode: Spark mode ('local' or 'cluster')
        partitions: Number of partitions
        
    Returns:
        Configured SparkSession
    """
    if mode == "local":
        master = f"local[{partitions}]"
    else:
        master = "yarn"
    
    builder = SparkSession.builder \
        .appName(SPARK_CONFIG["app_name"]) \
        .master(master)
    
    # Add configuration
    for key, value in SPARK_CONFIG.items():
        if key not in ["app_name", "master"]:
            builder = builder.config(key, value)
    
    # Set partitions
    builder = builder.config("spark.default.parallelism", str(partitions))
    builder = builder.config("spark.sql.shuffle.partitions", str(partitions * 50))
    
    # Additional Windows/Java 21+ compatibility settings
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Multimodal Twitter Data Analysis using PySpark"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Twitter data file (JSON format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to output directory for results"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "cluster"],
        help="Spark execution mode"
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=4,
        help="Number of Spark partitions"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Twitter Multimodal Analysis - PBDM Project")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Spark mode: {args.mode}")
    print(f"Partitions: {args.partitions}")
    print("=" * 60)
    
    # Create Spark session
    print("\n[1/5] Initializing Spark session...")
    spark = create_spark_session(args.mode, args.partitions)
    print("✓ Spark session created")
    
    try:
        # Data ingestion
        print("\n[2/5] Loading and preprocessing data...")
        ingestion = TwitterDataIngestion(spark)
        df = ingestion.load_json_data(str(input_path))
        print(f"✓ Loaded {df.count():,} tweets")
        
        df = ingestion.preprocess_data(df)
        print(f"✓ Preprocessed {df.count():,} tweets")
        
        # Display statistics
        stats = ingestion.get_statistics(df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Sentiment analysis
        print("\n[3/5] Performing sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer(spark)
        df = sentiment_analyzer.analyze_sentiment(df)
        print("✓ Sentiment analysis completed")
        
        # Display sentiment distribution
        sentiment_dist = df.groupBy("sentiment_label").count().orderBy("sentiment_label")
        print("\nSentiment Distribution:")
        for row in sentiment_dist.collect():
            print(f"  {row['sentiment_label']}: {row['count']:,}")
        
        # Factuality detection
        print("\n[4/5] Performing factuality detection...")
        factuality_detector = FactualityDetector(spark)
        df = factuality_detector.detect_factuality(df)
        print("✓ Factuality detection completed")
        
        # Display factuality distribution
        factuality_dist = df.groupBy("reliability_label").count().orderBy("reliability_label")
        print("\nFactuality Distribution:")
        for row in factuality_dist.collect():
            print(f"  {row['reliability_label']}: {row['count']:,}")
        
        # Visualization and reporting
        if not args.skip_visualization:
            print("\n[5/5] Generating visualizations and reports...")
            visualizer = Visualizer(spark, str(output_dir))
            
            # Generate all visualizations
            visualizer.plot_sentiment_distribution(df)
            visualizer.plot_factuality_distribution(df)
            visualizer.plot_sentiment_vs_factuality(df)
            visualizer.plot_engagement_analysis(df)
            
            # Generate word clouds
            visualizer.create_wordcloud(df)
            visualizer.create_wordcloud(df, sentiment="positive")
            visualizer.create_wordcloud(df, sentiment="negative")
            
            # Generate summary report
            visualizer.generate_summary_report(df)
            
            # Save results to CSV
            visualizer.save_results_csv(df)
            
            print("✓ All visualizations and reports generated")
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        spark.stop()
        print("\nSpark session stopped")


if __name__ == "__main__":
    main()
