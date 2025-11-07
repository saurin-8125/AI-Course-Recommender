"""
🎓 COMPLETE EXPLANATION OF ALL FUNCTIONS AND CONCEPTS
=====================================================

This file explains EVERY function, concept, and line of code in detail
so you can understand and explain everything about your AI course recommender.
"""

# ============================================================================
# PART 1: UNDERSTANDING BASIC CONCEPTS
# ============================================================================

def explain_vectors():
    """
    🧠 What is a Vector?

    A vector is just a LIST OF NUMBERS that represents something.
    Think of it like a recipe or a fingerprint made of numbers.
    """
    print("="*60)
    print("🔢 UNDERSTANDING VECTORS")
    print("="*60)

    # Example 1: Simple vector
    print("Example 1: Person's food preferences")
    print("Person A likes: Pizza=5, Burger=3, Salad=1")
    print("As a vector: [5, 3, 1]")
    print()

    # Example 2: Course as vector
    print("Example 2: Course description as vector")
    print("Course: 'Python programming for data analysis'")
    print("Important words: python=0.8, programming=0.7, data=0.9, analysis=0.6")
    print("As a vector: [0.8, 0.7, 0.9, 0.6]")
    print()

    print("💡 Key Point: Vectors let computers compare things using math!")
    print("   Similar vectors = Similar content")
    print("   Different vectors = Different content")

def explain_pandas():
    """
    🐼 What is pandas?

    pandas is a Python library for working with data in tables (like Excel).
    It's like having a super-powered Excel inside Python.
    """
    print("\n" + "="*60)
    print("🐼 UNDERSTANDING PANDAS")
    print("="*60)

    print("What pandas does:")
    print("✅ Read CSV files (like your courses.csv)")
    print("✅ Filter data (find courses in 'Business Finance')")
    print("✅ Sort data (arrange by similarity score)")
    print("✅ Select columns (get only title and duration)")
    print()

    print("Key pandas functions we use:")
    print("📖 pd.read_csv('file.csv') - Load data from CSV file")
    print("🔍 df[df['subject'] == 'AI'] - Filter rows where subject is AI")
    print("📊 df.sort_values('score') - Sort by a column")
    print("🎯 df.head(5) - Get first 5 rows")
    print("📋 df['column'].tolist() - Convert column to list")

def explain_tfidf_step_by_step():
    """
    🧠 TF-IDF Explained Step by Step

    TF-IDF converts text into numbers so computers can understand it.
    """
    print("\n" + "="*60)
    print("🧠 TF-IDF EXPLAINED STEP BY STEP")
    print("="*60)

    print("🎯 Goal: Convert text to numbers")
    print()

    print("Step 1: TF (Term Frequency)")
    print("   - Count how often each word appears in a document")
    print("   - 'Python appears 3 times in 10 words = 3/10 = 0.3'")
    print()

    print("Step 2: IDF (Inverse Document Frequency)")
    print("   - How rare is this word across ALL documents?")
    print("   - Common words (like 'the') get low scores")
    print("   - Rare words (like 'tensorflow') get high scores")
    print()

    print("Step 3: TF-IDF = TF × IDF")
    print("   - Words that are both frequent AND rare are important")
    print("   - Example: 'Python' appears often in a Python course (high TF)")
    print("   - But 'Python' doesn't appear in cooking courses (high IDF)")
    print("   - Result: 'Python' gets high TF-IDF score for Python courses")
    print()

    print("🔤 Example:")
    print("Course: 'Advanced Python Machine Learning'")
    print("TF-IDF might create: [advanced=0.4, python=0.8, machine=0.6, learning=0.7]")

def explain_cosine_similarity():
    """
    📐 Cosine Similarity Explained

    Measures how similar two vectors are by calculating the angle between them.
    """
    print("\n" + "="*60)
    print("📐 COSINE SIMILARITY EXPLAINED")
    print("="*60)

    print("🎯 Goal: Compare two vectors to see how similar they are")
    print()

    print("Think of vectors as arrows pointing in space:")
    print("   - Same direction = 100% similar (cosine = 1.0)")
    print("   - Opposite direction = 0% similar (cosine = 0.0)")
    print("   - Perpendicular = 50% similar (cosine = 0.5)")
    print()

    print("📊 Example:")
    print("User wants: [python=0.8, data=0.9, analysis=0.6]")
    print("Course A:   [python=0.7, data=0.8, analysis=0.5]")
    print("Course B:   [java=0.9, web=0.8, design=0.7]")
    print()
    print("Result:")
    print("   User vs Course A = 0.95 (95% similar) ✅")
    print("   User vs Course B = 0.12 (12% similar) ❌")
    print()
    print("💡 Course A is much more similar to what the user wants!")

# ============================================================================
# PART 2: UNDERSTANDING EVERY FUNCTION WE USE
# ============================================================================

def explain_sklearn_functions():
    """
    🔬 scikit-learn Functions Explained

    scikit-learn is a machine learning library. We use it for TF-IDF and similarity.
    """
    print("\n" + "="*60)
    print("🔬 SCIKIT-LEARN FUNCTIONS")
    print("="*60)

    print("1. TfidfVectorizer:")
    print("   What it does: Converts text to TF-IDF vectors")
    print("   Input: List of text ['course 1 description', 'course 2 description']")
    print("   Output: Matrix of numbers (vectors)")
    print()

    print("   Parameters:")
    print("   - stop_words='english': Remove common words ('the', 'and', 'is')")
    print("   - max_features=100: Keep only top 100 most important words")
    print("   - ngram_range=(1,2): Use single words AND two-word phrases")
    print()

    print("2. cosine_similarity:")
    print("   What it does: Compares two vectors")
    print("   Input: Two vectors or matrices")
    print("   Output: Similarity score (0.0 to 1.0)")
    print()

    print("3. fit_transform:")
    print("   What it does: Learn vocabulary AND convert text to vectors")
    print("   Step 1 (fit): Learn what words are important")
    print("   Step 2 (transform): Convert text to numbers")

def explain_python_functions():
    """
    🐍 Python Functions Explained

    Basic Python functions we use in our recommender.
    """
    print("\n" + "="*60)
    print("🐍 PYTHON FUNCTIONS EXPLAINED")
    print("="*60)

    print("1. ' '.join(list):")
    print("   What it does: Combine list items into one string")
    print("   Example: ['python', 'data'] → 'python data'")
    print()

    print("2. .fillna(''):")
    print("   What it does: Replace missing values with empty string")
    print("   Why: Prevents errors when some courses have no description")
    print()

    print("3. .tolist():")
    print("   What it does: Convert pandas column to Python list")
    print("   Example: DataFrame column → ['item1', 'item2', 'item3']")
    print()

    print("4. .flatten():")
    print("   What it does: Convert 2D array to 1D array")
    print("   Example: [[0.8, 0.9]] → [0.8, 0.9]")
    print()

    print("5. .copy():")
    print("   What it does: Create a copy of DataFrame")
    print("   Why: Prevents warnings when modifying data")
    print()

    print("6. enumerate(list, start):")
    print("   What it does: Add numbers to list items")
    print("   Example: ['a', 'b'] → [(1, 'a'), (2, 'b')]")

# ===========================================================================
# PART 3: COMPLETE COURSE RECOMMENDER SYSTEM
# ===========================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class CourseRecommender:
    """
    🎓 Complete Course Recommendation System

    This class contains everything needed for AI-powered course recommendations.
    """

    def __init__(self, data_path=None):
        """
        Initialize the recommender system.

        Args:
            data_path (str): Path to the CSV file containing course data
        """
        self.data_path = data_path or '/Users/dwijdesai/Desktop/AI Course Recommender /data/courses.csv'
        self.courses_df = None
        self.vectorizer = None
        self.tfidf_matrix = None

        print("🚀 Course Recommender System Initialized!")

    def load_data(self):
        """
        Load course data from CSV file.

        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.courses_df = pd.read_csv(self.data_path)
            print(f"✅ Successfully loaded {len(self.courses_df)} courses")

            # Show available subjects
            subjects = self.courses_df['subject'].unique()
            print(f"📚 Available subjects: {list(subjects)}")

            return True

        except FileNotFoundError:
            print(f"❌ Error: Could not find file at {self.data_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def prepare_tfidf_vectors(self, subject_filter=None):
        """
        Prepare TF-IDF vectors for all courses (or filtered by subject).

        Args:
            subject_filter (str): Filter courses by subject (optional)
        """
        if self.courses_df is None:
            print("❌ No data loaded. Call load_data() first.")
            return

        # Filter by subject if specified
        if subject_filter:
            df_filtered = self.courses_df[
                self.courses_df['subject'].str.lower() == subject_filter.lower()
            ]
            if df_filtered.empty:
                print(f"❌ No courses found for subject '{subject_filter}'")
                return
        else:
            df_filtered = self.courses_df

        # Get course descriptions
        course_texts = df_filtered['text_for_recommendations'].fillna('').tolist()

        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(course_texts)
        self.filtered_df = df_filtered

        print(f"🧠 Prepared TF-IDF vectors for {len(df_filtered)} courses")
        print(f"📊 Vector dimensions: {self.tfidf_matrix.shape}")

    def find_similar_courses(self, user_interests, subject=None, top_n=5):
        """
        Find courses similar to user interests using TF-IDF and cosine similarity.

        Args:
            user_interests (list): List of topics user is interested in
            subject (str): Subject to filter by (optional)
            top_n (int): Number of recommendations to return

        Returns:
            pandas.DataFrame: Top matching courses with similarity scores
        """
        print(f"🔍 Finding courses for interests: {user_interests}")
        if subject:
            print(f"📚 Filtering by subject: {subject}")

        # Load data if not already loaded
        if self.courses_df is None:
            if not self.load_data():
                return None

        # Prepare vectors for the specified subject
        self.prepare_tfidf_vectors(subject)

        if self.tfidf_matrix is None:
            return None

        # Convert user interests to query string
        user_query = ' '.join(user_interests)

        # Transform user query to TF-IDF vector
        user_vector = self.vectorizer.transform([user_query])

        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        # Add similarity scores to dataframe
        results_df = self.filtered_df.copy()
        results_df['similarity_score'] = similarities

        # Sort by similarity and return top matches
        top_matches = results_df.sort_values(
            'similarity_score',
            ascending=False
        ).head(top_n)

        return top_matches

    def display_recommendations(self, recommendations, user_interests):
        """
        Display course recommendations in a user-friendly format.

        Args:
            recommendations (pandas.DataFrame): Course recommendations
            user_interests (list): Original user interests
        """
        if recommendations is None or recommendations.empty:
            print("❌ No recommendations found.")
            return

        print("\n" + "="*70)
        print("🎓 AI-POWERED COURSE RECOMMENDATIONS")
        print("="*70)
        print(f"🎯 Based on your interests: {', '.join(user_interests)}")
        print()

        for idx, (_, course) in enumerate(recommendations.iterrows(), 1):
            similarity_percent = course['similarity_score'] * 100

            print(f"📚 {idx}. {course['course_title']}")
            print(f"   📊 Relevance Score: {similarity_percent:.1f}%")
            print(f"   ⏱️  Duration: {course['content_duration']:.1f} hours")
            print(f"   📈 Level: {course['level']}")
            print(f"   👥 Students: {course['num_subscribers']:,}")
            print(f"   ⭐ Reviews: {course['num_reviews']:,}")
            print()

        print("="*70)

    def get_course_statistics(self, subject=None):
        """
        Get statistics about the course dataset.

        Args:
            subject (str): Subject to get statistics for (optional)
        """
        if self.courses_df is None:
            self.load_data()

        df = self.courses_df
        if subject:
            df = df[df['subject'].str.lower() == subject.lower()]

        print(f"\n📊 COURSE STATISTICS")
        print(f"📚 Total courses: {len(df)}")
        print(f"⏱️  Average duration: {df['content_duration'].mean():.1f} hours")
        print(f"💰 Average price: ${df['price_usd'].mean():.2f}")
        print(f"👥 Average students: {df['num_subscribers'].mean():.0f}")
        print(f"📈 Level distribution:")
        for level, count in df['level'].value_counts().items():
            print(f"   {level}: {count} courses")

# ============================================================================
# PART 4: STUDY PLANNER INTEGRATION
# ============================================================================

from datetime import datetime, timedelta

class StudyPlanner:
    """
    📅 AI-Powered Study Planner

    Creates personalized study plans using AI course recommendations.
    """

    def __init__(self, recommender):
        """
        Initialize study planner with course recommender.

        Args:
            recommender (CourseRecommender): Course recommendation system
        """
        self.recommender = recommender

    def create_study_plan(self, user_interests, subject, exam_date, daily_hours=2):
        """
        Create a personalized study plan based on user interests and timeline.

        Args:
            user_interests (list): Topics user wants to learn
            subject (str): Subject area
            exam_date (str): Target date in 'YYYY-MM-DD' format
            daily_hours (int): Hours available per day

        Returns:
            dict: Complete study plan with recommended courses
        """
        # Calculate available time
        today = datetime.now()
        target_date = datetime.strptime(exam_date, '%Y-%m-%d')
        days_left = (target_date - today).days

        if days_left <= 0:
            return {"error": "Exam date must be in the future!"}

        total_hours = days_left * daily_hours

        print(f"📅 Creating study plan:")
        print(f"   🎯 Interests: {user_interests}")
        print(f"   📚 Subject: {subject}")
        print(f"   📆 Days left: {days_left}")
        print(f"   ⏰ Total hours: {total_hours}")

        # Get course recommendations
        recommendations = self.recommender.find_similar_courses(
            user_interests, subject, top_n=20
        )

        if recommendations is None or recommendations.empty:
            return {"error": "No courses found for your interests"}

        # Select courses that fit in available time
        selected_courses = []
        used_hours = 0

        for _, course in recommendations.iterrows():
            course_duration = course['content_duration']
            if used_hours + course_duration <= total_hours:
                selected_courses.append({
                    'title': course['course_title'],
                    'duration': course_duration,
                    'level': course['level'],
                    'similarity': course['similarity_score'],
                    'students': course['num_subscribers'],
                    'reviews': course['num_reviews']
                })
                used_hours += course_duration
            else:
                break

        # Calculate daily schedule
        if selected_courses:
            avg_hours_per_day = used_hours / days_left
            completion_date = today + timedelta(days=int(used_hours / daily_hours))
        else:
            avg_hours_per_day = 0
            completion_date = today

        return {
            'user_interests': user_interests,
            'subject': subject,
            'days_left': days_left,
            'total_hours': total_hours,
            'used_hours': used_hours,
            'selected_courses': selected_courses,
            'avg_hours_per_day': avg_hours_per_day,
            'completion_date': completion_date.strftime('%Y-%m-%d'),
            'schedule_efficiency': (used_hours / total_hours) * 100
        }

    def display_study_plan(self, plan):
        """
        Display study plan in a user-friendly format.

        Args:
            plan (dict): Study plan from create_study_plan()
        """
        if 'error' in plan:
            print(f"❌ Error: {plan['error']}")
            return

        print("\n" + "="*70)
        print("📅 PERSONALIZED AI STUDY PLAN")
        print("="*70)

        print(f"🎯 Your Interests: {', '.join(plan['user_interests'])}")
        print(f"📚 Subject: {plan['subject']}")
        print(f"📆 Days Available: {plan['days_left']}")
        print(f"⏰ Total Study Hours: {plan['total_hours']}")
        print(f"✅ Planned Hours: {plan['used_hours']:.1f}")
        print(f"📊 Schedule Efficiency: {plan['schedule_efficiency']:.1f}%")
        print(f"🏁 Completion Date: {plan['completion_date']}")
        print()

        print("📋 RECOMMENDED COURSES:")
        print("-" * 50)

        for i, course in enumerate(plan['selected_courses'], 1):
            print(f"{i}. {course['title']}")
            print(f"   📊 Relevance: {course['similarity']*100:.1f}%")
            print(f"   ⏱️  Duration: {course['duration']:.1f} hours")
            print(f"   📈 Level: {course['level']}")
            print(f"   👥 Students: {course['students']:,}")
            print()

        print("="*70)

# ============================================================================
# PART 5: TESTING AND DEMONSTRATION
# ============================================================================

def run_complete_demo():
    """
    🎬 Complete demonstration of the AI course recommender system.
    """
    print("🚀 COMPLETE AI COURSE RECOMMENDER DEMO")
    print("=" * 60)

    # Initialize system
    recommender = CourseRecommender()
    study_planner = StudyPlanner(recommender)

    # Demo 1: Basic course recommendations
    print("\n🎯 DEMO 1: Course Recommendations")
    print("-" * 40)

    interests = ['excel', 'financial', 'analysis']
    subject = 'Business Finance'

    recommendations = recommender.find_similar_courses(
        interests, subject, top_n=3
    )

    if recommendations is not None:
        recommender.display_recommendations(recommendations, interests)

    # Demo 2: Study plan creation
    print("\n🎯 DEMO 2: AI Study Plan")
    print("-" * 40)

    plan = study_planner.create_study_plan(
        user_interests=['investment', 'trading', 'portfolio'],
        subject='Business Finance',
        exam_date='2025-03-15',
        daily_hours=3
    )

    study_planner.display_study_plan(plan)

    # Demo 3: Different subject
    print("\n🎯 DEMO 3: Different Subject")
    print("-" * 40)

    web_interests = ['javascript', 'html', 'css', 'frontend']
    web_recommendations = recommender.find_similar_courses(
        web_interests, 'Web Development', top_n=3
    )

    if web_recommendations is not None:
        recommender.display_recommendations(web_recommendations, web_interests)

    # Demo 4: Statistics
    print("\n🎯 DEMO 4: Course Statistics")
    print("-" * 40)

    recommender.get_course_statistics('Business Finance')

if __name__ == "__main__":
    # First, explain all concepts
    print("🎓 LEARNING SESSION: Understanding Every Concept")
    print("=" * 60)

    explain_vectors()
    explain_pandas()
    explain_tfidf_step_by_step()
    explain_cosine_similarity()
    explain_sklearn_functions()
    explain_python_functions()

    # Then run the complete demo
    run_complete_demo()

    print("\n🎉 CONGRATULATIONS!")
    print("You now understand every concept and function in your AI course recommender!")
    print("You can explain TF-IDF, cosine similarity, vectors, and pandas to anyone!")
