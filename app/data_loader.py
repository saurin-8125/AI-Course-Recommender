# AI Course Recommender - Data Loader
# This file handles loading and cleaning our course data
# Every line is commented for maximum understanding

# ============================================================================
# IMPORTS - Libraries we need for data processing
# ============================================================================

# pandas: Used for data manipulation (reading CSV, filtering, sorting)
import pandas as pd

from .config import COURSES_FILE

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_courses(data_path=None):
    """
    Load course data from CSV file with basic error handling.

    This function:
    1. Reads the CSV file using pandas
    2. Shows basic information about the dataset
    3. Handles common errors

    Args:
        data_path (str): Path to the CSV file containing course data

    Returns:
        pandas.DataFrame: Loaded course data, or empty DataFrame if error
    """
    if data_path is None:
        data_path = COURSES_FILE

    # try: Start error handling block
    try:
        # pd.read_csv(): Pandas function to read CSV file into DataFrame
        # DataFrame is like a table with rows and columns
        df = pd.read_csv(data_path)

        # len(df): Count number of rows in DataFrame
        # f-string: Format string with variables inside {}
        print(f"✅ Successfully loaded {len(df)} courses from {data_path}")

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

def clean_data(df):
    """
    Clean and prepare the dataset for recommendations.

    This function:
    1. Creates a text column for recommendations
    2. Converts price to readable format
    3. Creates popularity scores
    4. Cleans level column

    Args:
        df (pandas.DataFrame): Raw course data

    Returns:
        pandas.DataFrame: Cleaned course data
    """

    # Check if DataFrame is empty
    if df.empty:
        print("❌ Cannot clean empty dataset")
        return df

    # .copy(): Create a copy to avoid modifying original data
    df_clean = df.copy()

    # Step 1: Create text column for recommendations
    # df['course_title']: Get course title column
    # df['subject']: Get subject column
    # +: Concatenate (join) two columns with space
    df_clean['text_for_recommendations'] = df_clean['course_title'] + ' ' + df_clean['subject']

    # Step 2: Convert price to readable format
    # Check if 'price' column exists
    if 'price' in df_clean.columns:
        # df['price'] / 100: Divide price by 100 (assuming price is in cents)
        df_clean['price_usd'] = df_clean['price'] / 100

    # Step 3: Create popularity score
    # Check if required columns exist
    if 'num_subscribers' in df_clean.columns and 'num_reviews' in df_clean.columns:
        # *: Multiply number of subscribers by number of reviews
        df_clean['popularity_score'] = df_clean['num_subscribers'] * df_clean['num_reviews']

    # Step 4: Clean level column
    # Check if 'level' column exists
    if 'level' in df_clean.columns:
        # .str.lower(): Convert all text in column to lowercase
        df_clean['level_clean'] = df_clean['level'].str.lower()

    # Print cleaning summary
    print("✅ Data cleaning completed:")
    print(f"   📝 Added text_for_recommendations column")

    if 'price_usd' in df_clean.columns:
        print(f"   💰 Converted price to USD format")

    if 'popularity_score' in df_clean.columns:
        print(f"   📊 Created popularity scores")

    if 'level_clean' in df_clean.columns:
        print(f"   🎯 Cleaned level column")

    return df_clean

def display_data_info(df):
    """
    Display useful information about the dataset.

    This function shows:
    1. Dataset overview (total courses, subjects, levels)
    2. Price and duration ranges
    3. Top popular courses

    Args:
        df (pandas.DataFrame): Course data to analyze
    """

    # Check if DataFrame is empty
    if df.empty:
        print("❌ Cannot display info for empty dataset")
        return

    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)

    # len(df): Count total number of courses
    print(f"📚 Total courses: {len(df)}")

    # Display available subjects
    if 'subject' in df.columns:
        # df['subject'].unique(): Get unique values from subject column
        subjects = df['subject'].unique()
        print(f"🎯 Subjects: {list(subjects)}")

    # Display available levels
    if 'level' in df.columns:
        # df['level'].unique(): Get unique values from level column
        levels = df['level'].unique()
        print(f"📈 Levels: {list(levels)}")

    # Display price range
    if 'price_usd' in df.columns:
        # .min(): Get minimum value in column
        # .max(): Get maximum value in column
        # :.2f: Format number to 2 decimal places
        min_price = df['price_usd'].min()
        max_price = df['price_usd'].max()
        print(f"💰 Price range: ${min_price:.2f} - ${max_price:.2f}")

    # Display duration range
    if 'content_duration' in df.columns:
        # :.1f: Format number to 1 decimal place
        min_duration = df['content_duration'].min()
        max_duration = df['content_duration'].max()
        print(f"⏱️  Duration range: {min_duration:.1f} - {max_duration:.1f} hours")

    # Display top popular courses
    if 'popularity_score' in df.columns:
        print(f"\n🏆 TOP 5 MOST POPULAR COURSES:")
        print("-" * 50)

        # .nlargest(5, 'popularity_score'): Get 5 rows with highest popularity scores
        # [columns]: Select specific columns to display
        top_courses = df.nlargest(5, 'popularity_score')[
            ['course_title', 'subject', 'num_subscribers', 'num_reviews']
        ]

        # .iterrows(): Iterate over DataFrame rows
        # enumerate(): Add numbers starting from 1
        for idx, (_, course) in enumerate(top_courses.iterrows(), 1):
            print(f"{idx}. {course['course_title']}")
            print(f"   📚 Subject: {course['subject']}")
            print(f"   👥 Students: {course['num_subscribers']:,}")
            print(f"   ⭐ Reviews: {course['num_reviews']:,}")
            print()

def get_subject_statistics(df, subject=None):
    """
    Get statistics for a specific subject or all subjects.

    Args:
        df (pandas.DataFrame): Course data
        subject (str): Specific subject to analyze (optional)
    """

    # Check if DataFrame is empty
    if df.empty:
        print("❌ Cannot get statistics for empty dataset")
        return

    # Filter by subject if specified
    if subject:
        # df['subject'].str.lower(): Convert subject column to lowercase
        # subject.lower(): Convert input subject to lowercase
        df_filtered = df[df['subject'].str.lower() == subject.lower()]

        if df_filtered.empty:
            print(f"❌ No courses found for subject '{subject}'")
            return

        print(f"\n📊 STATISTICS FOR '{subject.upper()}'")
    else:
        df_filtered = df
        print(f"\n📊 OVERALL STATISTICS")

    print("="*50)

    # Basic statistics
    print(f"📚 Total courses: {len(df_filtered)}")

    # Average duration
    if 'content_duration' in df_filtered.columns:
        # .mean(): Calculate average value
        avg_duration = df_filtered['content_duration'].mean()
        print(f"⏱️  Average duration: {avg_duration:.1f} hours")

    # Average price
    if 'price_usd' in df_filtered.columns:
        avg_price = df_filtered['price_usd'].mean()
        print(f"💰 Average price: ${avg_price:.2f}")

    # Average students
    if 'num_subscribers' in df_filtered.columns:
        avg_students = df_filtered['num_subscribers'].mean()
        print(f"👥 Average students: {avg_students:.0f}")

    # Level distribution
    if 'level' in df_filtered.columns:
        print(f"\n📈 Level distribution:")
        # .value_counts(): Count occurrences of each unique value
        # .items(): Get key-value pairs
        for level, count in df_filtered['level'].value_counts().items():
            print(f"   {level}: {count} courses")

# ============================================================================
# MAIN EXECUTION AND TESTING
# ============================================================================

def test_data_loading():
    """
    Test all data loading and cleaning functions.
    """

    print("🧪 Testing Data Loading System")
    print("="*50)

    # Test 1: Load data
    print("\n📋 Test 1: Loading data...")
    df = load_courses()

    if df.empty:
        print("❌ Cannot proceed without data")
        return

    # Test 2: Clean data
    print("\n📋 Test 2: Cleaning data...")
    df_clean = clean_data(df)

    # Test 3: Display information
    print("\n📋 Test 3: Displaying data info...")
    display_data_info(df_clean)

    # Test 4: Subject statistics
    print("\n📋 Test 4: Subject statistics...")
    get_subject_statistics(df_clean, "Business Finance")

    print("\n✅ All tests completed!")

# if __name__ == "__main__": Only run this code when file is executed directly
if __name__ == "__main__":

    # Run all tests
    test_data_loading()
