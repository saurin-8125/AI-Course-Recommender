# AI Course Recommender - Topic Matcher
# This file contains our TF-IDF course recommendation system

# ============================================================================
# IMPORTS - Libraries we need for our AI system
# ============================================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import COURSES_FILE, TFIDF_CONFIG, RECOMMENDATION_CONFIG

# Remove unused import that was causing issues
# from streamlit.elements.iframe import IframeMixin

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_course_data(data_path=None):
    """Load course data from CSV file with error handling."""
    if data_path is None:
        data_path = COURSES_FILE

    # try: Start error handling block
    try:
        df = pd.read_csv(data_path)

        print(f"✅ Successfully loaded {len(df)} courses from {data_path}")

        subjects = df['subject'].unique()

        # list(): Convert numpy array to Python list for display
        print(f"📚 Available subjects: {list(subjects)}")

        # return: Send the DataFrame back to whoever called this function
        return df

    # except: Handle specific error when file is not found
    except FileNotFoundError:
        print(f"❌ Error: Could not find file at {data_path}")
        print("   Please check if the file exists and the path is correct.")

        # pd.DataFrame(): Create empty DataFrame when error occurs
        return pd.DataFrame()

    # except: Handle any other errors
    except Exception as e:
        # str(e): Convert error message to string
        print(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# CORE RECOMMENDATION FUNCTIONS
# ============================================================================

def find_best_courses(user_topics, subject, top_n=None):
    """Find best course recommendations using TF-IDF and cosine similarity."""
    if top_n is None:
        top_n = RECOMMENDATION_CONFIG['default_top_n']

    print(f"🔍 Searching for courses in '{subject}' related to: {user_topics}")

    # Step 1: Load course data using our function
    df = load_course_data()

    # Check if the DataFrame is empty
    if df.empty:
        print("❌ Could not load course data.")
        return pd.DataFrame()

    # Step 2: Filter courses by subject
    df_filtered = df[df['subject'].str.lower() == subject.lower()]

    # Check if any courses match the subject
    if df_filtered.empty:
        # df['subject'].unique(): Get all available subjects
        available_subjects = df['subject'].unique()
        print(f"❌ No courses found for subject '{subject}'.")
        print(f"Available subjects: {list(available_subjects)}")
        return pd.DataFrame()

    # Print how many courses we found
    print(f"📚 Found {len(df_filtered)} courses for: '{subject}'")

    # Step 3: Prepare text data for TF-IDF
    course_texts = df_filtered['text_for_recommendations'].fillna('').tolist() # this is list of course texts

    # Step 4: Convert user topics to search query
    user_query = ' '.join(user_topics)
    print(f"🎯 User query: '{user_query}'")

    # Add user query to course texts for TF-IDF processing
    all_texts = course_texts + [user_query]

    # Step 5: Create TF-IDF vectors
    print("🧠 Converting text to TF-IDF vectors...")
    # TfidfVectorizer: Create TF-IDF converter
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)

    # .fit_transform(): Learn vocabulary AND convert text to vectors
    # Returns sparse matrix (efficient storage of mostly-zero numbers)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Print matrix dimensions for debugging
    print(f"✅ Created TF-IDF matrix with shape: {tfidf_matrix.shape}")

    # Step 6: Calculate similarity scores
    # tfidf_matrix[-1]: Get last row (user query vector)
    user_vector = tfidf_matrix[-1]

    # tfidf_matrix[:-1]: Get all rows except last (course vectors)
    course_vectors = tfidf_matrix[:-1]

    # cosine_similarity(): Calculate similarity between user and each course
    # .flatten(): Convert 2D array to 1D array
    similarities = cosine_similarity(user_vector, course_vectors).flatten()

    # Step 7: Add similarity scores to DataFrame
    # .copy(): Create copy to avoid warnings
    df_with_scores = df_filtered.copy()
    df_with_scores['similarity_score'] = similarities

    # Step 8: Sort and return top matches
    # .sort_values(): Sort DataFrame by column
    # ascending=False: Sort from highest to lowest
    # .head(top_n): Get first top_n rows
    top_courses = df_with_scores.sort_values('similarity_score', ascending=False).head(top_n)

    # Print summary
    print(f"🏆 Found {len(top_courses)} matching courses")

    # Select relevant columns to return
    result_columns = [
        'course_title',          # Course name
        'similarity_score',      # How well it matches (0-1)
        'content_duration',      # How long it takes
        'level',                 # Difficulty level
        'num_subscribers',       # How many students
        'num_reviews',           # Number of reviews
        'price_usd'             # Price in USD
    ]

    # Filter columns that actually exist in the DataFrame
    available_columns = [col for col in result_columns if col in top_courses.columns]

    # Return DataFrame with selected columns
    return top_courses[available_columns]

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_recommendations(recommendations, user_topics):
    """
    Display course recommendations in a user-friendly format.

    Args:
        recommendations (pandas.DataFrame): Course recommendations with scores
        user_topics (list): Original user interests
    """

    # Check if we have any recommendations
    if recommendations.empty:
        print("❌ No recommendations found.")
        return

    # Print header
    print("\n" + "="*70)
    print("🎓 AI-POWERED COURSE RECOMMENDATIONS")
    print("="*70)

    # ', '.join(): Convert list to comma-separated string
    print(f"🎯 Based on your interests: {', '.join(user_topics)}")
    print()

    # Loop through each recommended course
    # .iterrows(): Iterate over DataFrame rows
    # enumerate(): Add numbers starting from 1
    for idx, (_, course) in enumerate(recommendations.iterrows(), 1):

        # Get similarity score and convert to percentage
        similarity_percent = course['similarity_score'] * 100

        # Print course details
        print(f"📚 {idx}. {course['course_title']}")
        print(f"   📊 Relevance Score: {similarity_percent:.1f}%")

        # Check if column exists before accessing
        if 'content_duration' in course:
            print(f"   ⏱️  Duration: {course['content_duration']:.1f} hours")

        if 'level' in course:
            print(f"   📈 Level: {course['level']}")

        if 'num_subscribers' in course:
            # :, adds commas to large numbers (e.g., 1,234)
            print(f"   👥 Students: {course['num_subscribers']:,}")

        if 'num_reviews' in course:
            print(f"   ⭐ Reviews: {course['num_reviews']:,}")

        if 'price_usd' in course:
            print(f"   💰 Price: ${course['price_usd']:.2f}")

        print()  # Empty line between courses

    print("="*70)

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_recommendation_system():
    """
    Test the recommendation system with sample data.
    """

    print("🧪 Testing AI Course Recommendation System")
    print("="*50)

    # Test Case 1: Business Finance courses
    print("\n📋 Test Case 1: Business Finance")
    test_topics = ["excel", "financial", "analysis"]
    test_subject = "Business Finance"

    # Call our recommendation function
    results = find_best_courses(test_topics, test_subject, top_n=3)

    # Display results
    if not results.empty:
        display_recommendations(results, test_topics)

    # Test Case 2: Web Development courses
    print("\n📋 Test Case 2: Web Development")
    test_topics2 = ["javascript", "html", "css"]
    test_subject2 = "Web Development"

    results2 = find_best_courses(test_topics2, test_subject2, top_n=3)

    if not results2.empty:
        display_recommendations(results2, test_topics2)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# if __name__ == "__main__": Only run this code when file is executed directly
if __name__ == "__main__":

    # Test data loading first
    print("🔧 Testing data loading...")
    df = load_course_data()

    if not df.empty:
        print("✅ Data loading successful!")
        print(f"📊 Dataset shape: {df.shape}")
        print()

        # Run full recommendation system test
        test_recommendation_system()

    else:
        print("❌ Cannot proceed without data. Please check your CSV file.")
