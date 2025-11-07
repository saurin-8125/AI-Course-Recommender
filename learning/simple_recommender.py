"""
🎓 SIMPLE COURSE RECOMMENDER - LEARNING VERSION
==================================================

This file teaches you TF-IDF and cosine similarity step by step.
We'll start with a tiny example you can understand completely.

Goal: Learn how AI finds similar courses to what you want to study.
"""

# Step 1: Import the libraries we need
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def explain_tfidf_basics():
    """
    🧠 LESSON 1: Understanding TF-IDF with a tiny example

    Let's start with just 3 courses to understand the concept
    """
    print("="*60)
    print("🎯 LESSON 1: TF-IDF Basics")
    print("="*60)

    # Our tiny course database
    courses = [
        "Python programming for beginners with data analysis",
        "Advanced Python machine learning and AI",
        "Java programming fundamentals and web development"
    ]

    course_names = [
        "Python Data Course",
        "Python AI Course",
        "Java Web Course"
    ]

    print("📚 Our 3 sample courses:")
    for i, (name, desc) in enumerate(zip(course_names, courses)):
        print(f"{i+1}. {name}: '{desc}'")
        print(f"   📖 Description: {desc}")
        print(f"   📚 Length: {len(desc)} words")

    print("\n🔍 What happens when we search for: 'Python programming'")

    # Add user query to our list
    user_query = "Python programming"
    all_text = courses + [user_query]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_text)

    # Show what words TF-IDF found important
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n🔤 Important words TF-IDF found: {list(feature_names)}")

    # Calculate similarity between user query and each course
    user_vector = tfidf_matrix[-1]  # Last item is user query
    course_vectors = tfidf_matrix[:-1]  # All except last are courses

    similarities = cosine_similarity(user_vector, course_vectors).flatten()

    print("\n📊 Similarity Scores:")
    for i, (name, score) in enumerate(zip(course_names, similarities)):
        percentage = score * 100
        print(f"{i+1}. {name}: {percentage:.1f}% match")

    # Find the best match
    best_match_idx = np.argmax(similarities)
    print(f"\n🏆 Best match: {course_names[best_match_idx]}")
    print(f"   Why? Both have 'Python' and 'programming' words!")

    return similarities

def load_real_courses():
    """
    🧠 LESSON 2: Load your actual course data

    Now let's work with your real course data
    """
    print("\n" + "="*60)
    print("🎯 LESSON 2: Working with Real Data")
    print("="*60)

    try:
        # Load your course data
        df = pd.read_csv('../data/courses.csv')
        print(f"✅ Loaded {len(df)} real courses!")

        # Show what subjects we have
        subjects = df['subject'].unique()
        print(f"📚 Available subjects: {list(subjects)}")

        # Focus on one subject for learning
        subject = "Business Finance"
        df_finance = df[df['subject'] == subject]
        print(f"\n🏦 Found {len(df_finance)} courses in '{subject}'")

        # Show first few course titles
        print("\n📋 Sample course titles:")
        for i, title in enumerate(df_finance['course_title'].head(5)):
            print(f"{i+1}. {title}")

        return df_finance

    except FileNotFoundError:
        print("❌ Could not find courses.csv file")
        print("   Make sure your data file exists!")
        return None

def simple_course_finder(user_interests, df, top_n=3):
    """
    🧠 LESSON 3: Build a simple course finder

    This function finds courses that match what you're interested in

    Args:
        user_interests (list): What you want to learn, e.g., ["excel", "finance"]
        df (DataFrame): Course data
        top_n (int): How many recommendations to return
    """
    print("\n" + "="*60)
    print("🎯 LESSON 3: Finding Matching Courses")
    print("="*60)

    print(f"🔍 You're interested in: {user_interests}")

    # Step 1: Get course descriptions
    # We'll use the 'text_for_recommendations' column which combines title + subject
    course_texts = df['text_for_recommendations'].fillna('').tolist()

    # Step 2: Convert your interests to a search query
    user_query = " ".join(user_interests)
    print(f"🎯 Search query: '{user_query}'")

    # Step 3: Add user query to the text list
    all_texts = course_texts + [user_query]

    # Step 4: Convert all text to TF-IDF vectors
    print("\n🧠 Converting text to numbers using TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words='english',  # Remove words like 'the', 'and', 'is'
        max_features=100       # Keep only top 100 most important words
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    print(f"✅ Created {tfidf_matrix.shape[0]} vectors with {tfidf_matrix.shape[1]} features each")

    # Step 5: Calculate how similar each course is to your interests
    user_vector = tfidf_matrix[-1]      # Your interests (last item)
    course_vectors = tfidf_matrix[:-1]   # All courses (everything except last)

    similarities = cosine_similarity(user_vector, course_vectors).flatten()

    # Step 6: Add similarity scores to the dataframe
    df_with_scores = df.copy()
    df_with_scores['similarity_score'] = similarities

    # Step 7: Sort by similarity and get top matches
    top_courses = df_with_scores.sort_values('similarity_score', ascending=False).head(top_n)

    # Step 8: Show results
    print(f"\n🏆 Top {top_n} course recommendations:")
    print("-" * 50)

    for i, (_, course) in enumerate(top_courses.iterrows(), 1):
        similarity_percent = course['similarity_score'] * 100
        print(f"{i}. {course['course_title']}")
        print(f"   📊 Match: {similarity_percent:.1f}%")
        print(f"   ⏱️  Duration: {course['content_duration']:.1f} hours")
        print(f"   📚 Level: {course['level']}")
        print()

    return top_courses

def explain_why_it_works():
    """
    🧠 LESSON 4: Why does this AI approach work?
    """
    print("="*60)
    print("🎯 LESSON 4: Why This AI Approach Works")
    print("="*60)

    print("🤔 Traditional approach:")
    print("   - Search for exact keyword matches")
    print("   - If you search 'excel', only find courses with 'excel' in title")
    print("   - Miss relevant courses that say 'spreadsheet' instead")

    print("\n🧠 AI approach (TF-IDF + Cosine Similarity):")
    print("   - Understand that 'excel' and 'spreadsheet' are related")
    print("   - Find courses about 'financial modeling' when you search 'finance'")
    print("   - Rank courses by how well they match your interests")

    print("\n🔍 How TF-IDF helps:")
    print("   - TF: Words that appear often in a course are important")
    print("   - IDF: Words that are rare across all courses are distinctive")
    print("   - Result: Finds courses with words that are both relevant AND unique")

    print("\n📐 How Cosine Similarity helps:")
    print("   - Treats each course as a 'direction' in word-space")
    print("   - Your interests also become a 'direction'")
    print("   - Finds courses pointing in the same direction as your interests")
    print("   - Score: 0% = completely different, 100% = identical")

# Main learning function
def run_learning_session():
    """
    🎓 Complete learning session - run all lessons
    """
    print("🚀 WELCOME TO AI COURSE RECOMMENDATION LEARNING!")
    print("We'll learn step by step how AI finds the best courses for you.\n")

    # Lesson 1: Basic TF-IDF with tiny example
    explain_tfidf_basics()

    # Lesson 2: Load real data
    df = load_real_courses()

    if df is not None and not df.empty:
        # Lesson 3: Find courses with real data
        user_interests = ["excel", "financial", "analysis"]
        simple_course_finder(user_interests, df, top_n=3)

        # Try another search
        print("\n" + "="*60)
        print("🔄 Let's try another search!")
        user_interests2 = ["investment", "trading", "stock"]
        simple_course_finder(user_interests2, df, top_n=3)

    # Lesson 4: Explain the theory
    explain_why_it_works()

    print("\n" + "="*60)
    print("🎉 CONGRATULATIONS!")
    print("You now understand how AI-powered course recommendation works!")
    print("="*60)

# Interactive testing function
def test_your_interests():
    """
    🧪 Test the recommender with your own interests
    """
    print("\n🧪 TEST YOUR OWN INTERESTS")
    print("="*40)

    # Load data
    try:
        df = pd.read_csv('../data/courses.csv')
        subjects = df['subject'].unique()

        print(f"Available subjects: {list(subjects)}")

        # You can modify these to test different interests
        my_interests = ["python", "data", "analysis"]  # <-- Change this!
        my_subject = "Business Finance"                # <-- Change this!

        print(f"\n🎯 Testing interests: {my_interests}")
        print(f"📚 In subject: {my_subject}")

        # Filter by subject
        df_subject = df[df['subject'] == my_subject]

        if not df_subject.empty:
            results = simple_course_finder(my_interests, df_subject, top_n=5)
            print("✅ Test completed!")
        else:
            print(f"❌ No courses found in subject '{my_subject}'")
            print(f"Available subjects: {list(subjects)}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Run when file is executed
if __name__ == "__main__":
    # Run the complete learning session
    run_learning_session()

    # Test with your own interests
    test_your_interests()
