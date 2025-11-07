# AI Course Recommender - Streamlit Web Interface
# This file creates a beautiful web interface for our course recommender
# Every line is commented for maximum understanding

# ============================================================================
# IMPORTS - Libraries we need for web interface

# streamlit: Used for creating web applications in Python
# os: Used for file path operations
import os

# sys: Used for system-specific parameters and functions
import sys

# datetime: Used for date calculations and formatting
from datetime import datetime, timedelta

# pandas: Used for data manipulation (reading CSV, filtering, sorting)
import pandas as pd
import streamlit as st

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from app.data_loader import display_data_info, load_courses
from app.study_planner import create_study_plan, display_study_plan
from app.topic_matcher import display_recommendations, find_best_courses

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# st.set_page_config(): Configure the Streamlit page
st.set_page_config(
    page_title="AI Course Recommender",  # Browser tab title
    page_icon="🎓",  # Browser tab icon
    layout="wide",  # Use wide layout for more space
    initial_sidebar_state="expanded",  # Start with sidebar open
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_data_cached():
    """
    Load course data with caching for better performance.

    @st.cache_data: Streamlit decorator to cache function results
    This means the data is loaded once and reused for better performance
    """

    # Try to load data from our data_loader module
    try:
        df = load_courses()
        return df
    except Exception as e:
        # st.error(): Display error message in red
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


# @st.cache_data: Cache this function to avoid reloading data
@st.cache_data
def get_available_subjects():
    """
    Get list of available subjects from the dataset.

    Returns:
        list: Available subjects in the dataset
    """

    df = load_data_cached()

    if df.empty:
        return []

    # df['subject'].unique(): Get unique subjects
    # .tolist(): Convert to Python list
    return df["subject"].unique().tolist()


def display_course_card(course, index):
    """
    Display a single course as a card with all relevant information.

    Args:
        course (pandas.Series): Course data
        index (int): Course number for display
    """

    # st.container(): Create a container for the course card
    with st.container():
        # st.markdown(): Display formatted text
        st.markdown(f"### 📚 {index}. {course['course_title']}")

        # Create columns for organized layout
        # st.columns(): Create columns for side-by-side layout
        col1, col2, col3 = st.columns(3)

        # Display course information in columns
        with col1:
            # Get similarity score and convert to percentage
            similarity_percent = course["similarity_score"] * 100
            st.metric(label="📊 Relevance Score", value=f"{similarity_percent:.1f}%")

            # st.metric(): Display a metric with label and value
            if "content_duration" in course:
                st.metric(
                    label="⏱️ Duration", value=f"{course['content_duration']:.1f} hours"
                )

        with col2:
            if "level" in course:
                st.metric(label="📈 Level", value=course["level"])

            if "num_subscribers" in course:
                st.metric(label="👥 Students", value=f"{course['num_subscribers']:,}")

        with col3:
            if "num_reviews" in course:
                st.metric(label="⭐ Reviews", value=f"{course['num_reviews']:,}")

            if "price_usd" in course:
                st.metric(label="💰 Price", value=f"${course['price_usd']:.2f}")

        # st.divider(): Add a horizontal line separator
        st.divider()


def display_study_plan_ui(plan):
    """
    Display study plan in Streamlit UI format.

    Args:
        plan (dict): Study plan data
    """

    # Check if there was an error
    if "error" in plan:
        st.error(plan["error"])
        return

    # Display plan overview
    st.subheader("📅 Your Personalized Study Plan")

    # Create metrics for key information
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📅 Days Available", plan["days_left"])

    with col2:
        st.metric("⏰ Daily Hours", plan["daily_hours"])

    with col3:
        st.metric("📊 Total Hours", plan["total_hours"])

    with col4:
        st.metric("🎯 Efficiency", f"{plan['schedule_efficiency']:.1f}%")

    # Display selected courses
    st.subheader("📋 Recommended Courses")

    if plan["selected_courses"]:
        # Loop through selected courses
        for i, course in enumerate(plan["selected_courses"], 1):
            # st.expander(): Create expandable section
            with st.expander(f"📚 {i}. {course['title']}", expanded=True):
                # Create columns for course details
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"📊 **Relevance:** {course['similarity'] * 100:.1f}%")
                    st.write(f"⏱️ **Duration:** {course['duration']:.1f} hours")

                with col2:
                    st.write(f"📈 **Level:** {course['level']}")
                    st.write(f"👥 **Students:** {course['students']:,}")

                with col3:
                    st.write(f"⭐ **Reviews:** {course['reviews']:,}")
                    if course.get("is_partial", False):
                        st.warning("⚠️ Partial completion due to time constraints")

    else:
        st.warning("❌ No courses selected (not enough time)")

    # Display daily schedule
    if plan.get("daily_schedule"):
        st.subheader("📅 Daily Schedule")

        # Convert daily schedule to DataFrame for better display
        schedule_df = pd.DataFrame(plan["daily_schedule"])

        # st.dataframe(): Display DataFrame as interactive table
        st.dataframe(
            schedule_df,
            use_container_width=True,  # Use full width
            hide_index=True,  # Hide row index
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """
    Main function that creates the Streamlit web interface.
    """

    # Application title
    st.title("🎓 AI Course Recommender")
    st.markdown("### Find the perfect courses for your learning journey using AI!")

    # Load data at start
    df = load_data_cached()

    if df.empty:
        st.error("❌ Could not load course data. Please check your data file.")
        st.stop()  # Stop execution if no data

    # Get available subjects
    subjects = get_available_subjects()

    # ============================================================================
    # SIDEBAR - User Input Section
    # ============================================================================

    st.sidebar.header("🎯 Your Learning Preferences")

    # Subject selection
    # st.sidebar.selectbox(): Create dropdown in sidebar
    selected_subject = st.sidebar.selectbox(
        "📚 Choose your subject area:",
        subjects,
        help="Select the subject you want to study",
    )

    # Topics input
    # st.sidebar.text_area(): Create text input area
    topics_input = st.sidebar.text_area(
        "🔍 What topics are you interested in?",
        placeholder="Enter topics separated by commas (e.g., excel, finance, analysis)",
        help="Enter the specific topics you want to learn about",
    )

    # Convert topics input to list
    if topics_input:
        # .split(','): Split string by commas
        # .strip(): Remove whitespace from each topic
        user_topics = [topic.strip() for topic in topics_input.split(",")]
    else:
        user_topics = []

    # Study planning inputs
    st.sidebar.header("📅 Study Planning")

    # Date input
    # st.sidebar.date_input(): Create date picker
    exam_date = st.sidebar.date_input(
        "🎯 Target exam/completion date:",
        min_value=datetime.now().date(),  # Can't select past dates
        value=datetime.now().date() + timedelta(days=30),  # Default to 30 days from now
        help="Select your target completion date",
    )

    # Hours per day input
    # st.sidebar.slider(): Create slider input
    daily_hours = st.sidebar.slider(
        "⏰ Hours available per day:",
        min_value=1,
        max_value=12,
        value=2,
        help="How many hours can you study per day?",
    )

    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "📊 Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many course recommendations do you want?",
    )

    # ============================================================================
    # MAIN CONTENT AREA
    # ============================================================================

    # Create tabs for different sections
    # st.tabs(): Create tabs for organizing content
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Course Recommendations", "📅 Study Plan", "📊 Dataset Info"]
    )

    # Tab 1: Course Recommendations
    with tab1:
        st.header("🎯 AI-Powered Course Recommendations")

        # Show search button
        if st.button("🔍 Find Courses", type="primary"):
            # Check if user entered topics
            if not user_topics:
                st.warning("⚠️ Please enter some topics you're interested in!")
            else:
                # Show what we're searching for
                st.info(
                    f"🔍 Searching for '{selected_subject}' courses related to: {', '.join(user_topics)}"
                )

                # Create progress bar
                # st.progress(): Show progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()  # Placeholder for status updates

                # Update progress
                progress_bar.progress(25)
                status_text.text("Loading course data...")

                # Get recommendations
                try:
                    progress_bar.progress(50)
                    status_text.text("Analyzing courses with AI...")

                    # Call our AI recommendation function
                    recommendations = find_best_courses(
                        user_topics, selected_subject, top_n=num_recommendations
                    )

                    progress_bar.progress(100)
                    status_text.text("Complete!")

                    # Display results
                    if not recommendations.empty:
                        st.success(f"✅ Found {len(recommendations)} matching courses!")

                        # Display each course
                        for idx, (_, course) in enumerate(
                            recommendations.iterrows(), 1
                        ):
                            display_course_card(course, idx)

                    else:
                        st.warning("❌ No courses found matching your criteria.")

                except Exception as e:
                    st.error(f"❌ Error getting recommendations: {str(e)}")

                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

    # Tab 2: Study Plan
    with tab2:
        st.header("📅 Personalized Study Plan")

        # Show create plan button
        if st.button("📅 Create Study Plan", type="primary"):
            # Check if user entered topics
            if not user_topics:
                st.warning("⚠️ Please enter some topics you're interested in!")
            else:
                # Show what we're planning for
                st.info(
                    f"📅 Creating study plan for '{selected_subject}' focusing on: {', '.join(user_topics)}"
                )

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    progress_bar.progress(33)
                    status_text.text("Getting course recommendations...")

                    # Convert date to string format
                    exam_date_str = exam_date.strftime("%Y-%m-%d")

                    progress_bar.progress(66)
                    status_text.text("Creating personalized study plan...")

                    # Create study plan
                    plan = create_study_plan(
                        user_topics=user_topics,
                        subject=selected_subject,
                        exam_date_str=exam_date_str,
                        daily_hours=daily_hours,
                    )

                    progress_bar.progress(100)
                    status_text.text("Study plan ready!")

                    # Display study plan
                    display_study_plan_ui(plan)

                except Exception as e:
                    st.error(f"❌ Error creating study plan: {str(e)}")

                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

    # Tab 3: Dataset Information
    with tab3:
        st.header("📊 Dataset Information")

        # Show dataset overview
        st.subheader("📋 Dataset Overview")

        # Display basic statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📚 Total Courses", len(df))

        with col2:
            st.metric("🎯 Subjects", len(subjects))

        with col3:
            if "content_duration" in df.columns:
                avg_duration = df["content_duration"].mean()
                st.metric("⏱️ Avg Duration", f"{avg_duration:.1f} hours")

        # Show subject distribution
        st.subheader("📊 Courses by Subject")

        if "subject" in df.columns:
            # Create bar chart of subject distribution
            subject_counts = df["subject"].value_counts()

            # st.bar_chart(): Create bar chart
            st.bar_chart(subject_counts)

        # Show sample data
        st.subheader("📋 Sample Course Data")

        # Display first few rows of data
        # st.dataframe(): Show interactive table
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # ============================================================================
    # FOOTER
    # ============================================================================

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666;'>
            <p>🎓 AI Course Recommender | Built with Streamlit and scikit-learn</p>
            <p>Find the perfect courses for your learning journey!</p>
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML in markdown
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

# if __name__ == "__main__": Only run this code when file is executed directly
if __name__ == "__main__":
    # Run the main Streamlit application
    main()
