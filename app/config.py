"""
AI Course Recommender - Configuration Management
Centralized configuration for all application settings
"""

from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
COURSES_FILE = DATA_DIR / "courses_large_deduplicated.csv"
# Use the deduplicated version as primary

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================

# TF-IDF Vectorizer settings
TFIDF_CONFIG = {
    "stop_words": "english",
    "max_features": 1000,
    "ngram_range": (1, 2),  # Single words and bigrams
    "max_df": 0.95,  # Ignore terms that appear in more than 95% of documents
    "min_df": 2,  # Ignore terms that appear in less than 2 documents
}

# Recommendation settings
RECOMMENDATION_CONFIG = {
    "default_top_n": 5,
    "min_similarity_threshold": 0.1,  # Minimum similarity score to consider
    "max_recommendations_for_planning": 20,  # More courses for study planning
}

# ============================================================================
# STUDY PLANNING CONFIGURATION
# ============================================================================

STUDY_CONFIG = {
    "default_daily_hours": 2,
    "min_daily_hours": 1,
    "max_daily_hours": 12,
    "min_days_ahead": 1,  # Minimum days before exam
    "max_days_ahead": 365,  # Maximum days for planning
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    "page_title": "AI Course Recommender",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

VALIDATION_CONFIG = {
    "max_topics_per_request": 10,
    "max_topic_length": 50,
    "allowed_subjects": [
        "Business Finance",
        "Web Development",
        "Graphic Design",
        "Musical Instruments",
    ],
    "date_format": "%Y-%m-%d",
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log",
}

# ============================================================================
# CHATBOT CONFIGURATION
# ============================================================================

CHATBOT_CONFIG = {
    "api_key": "YOUR_OPENAI_API_KEY",
    "model": "gpt-3.5-turbo",
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    "tfidf_cache_enabled": True,
    "cache_dir": PROJECT_ROOT / "cache",
    "cache_expiry_hours": 24,  # Cache TF-IDF matrices for 24 hours
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        DATA_DIR,
        PROJECT_ROOT / "logs",
        CACHE_CONFIG["cache_dir"],
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_data_file_path(filename: str) -> Path:
    """Get full path for a data file."""
    return DATA_DIR / filename


def validate_config():
    """Validate configuration settings."""
    # Check if data file exists
    if not COURSES_FILE.exists():
        msg = f"Required data file not found: {COURSES_FILE}"
        raise FileNotFoundError(msg)

    # Validate TF-IDF settings
    if TFIDF_CONFIG["max_features"] <= 0:
        raise ValueError("max_features must be positive")

    if TFIDF_CONFIG["ngram_range"][0] > TFIDF_CONFIG["ngram_range"][1]:
        msg = "ngram_range must be (min_n, max_n) with min_n <= max_n"
        raise ValueError(msg)

    print("✅ Configuration validated successfully")


# Initialize on import
ensure_directories()

if __name__ == "__main__":
    validate_config()
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Courses file: {COURSES_FILE}")
