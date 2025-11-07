# AI Course Recommender - Study Planner
# This file creates personalized study plans using AI recommendations
# Every line is commented for maximum understanding

# ===========================================================================
# IMPORTS - Libraries we need for study planning
# ===========================================================================

# pandas: Used for data manipulation (reading CSV, filtering, sorting)
import pandas as pd

# datetime: Used for date calculations and formatting
from datetime import datetime, timedelta

# Import our topic matcher for AI recommendations
from .topic_matcher import find_best_courses
from .config import STUDY_CONFIG, RECOMMENDATION_CONFIG

# ===========================================================================
# STUDY PLANNING FUNCTIONS
# ===========================================================================

def create_study_plan(user_topics, subject, exam_date_str,
                      daily_hours=None, today_str=None):
    """
    Create a personalized study plan using AI course recommendations.

    This function:
    1. Calculates available study time
    2. Gets AI-powered course recommendations
    3. Selects courses that fit in available time
    4. Creates a detailed study schedule

    Args:
        user_topics (list): Topics user wants to learn, e.g., ["excel", "finance"]
        subject (str): Subject area, e.g., "Business Finance"
        exam_date_str (str): Target exam date in 'YYYY-MM-DD' format
        daily_hours (int): Hours available per day for studying
        today_str (str): Current date in 'YYYY-MM-DD' format (optional)

    Returns:
        dict: Complete study plan with courses and schedule
    """

    if daily_hours is None:
        daily_hours = STUDY_CONFIG['default_daily_hours']

    print(f"📅 Creating study plan for: {user_topics}")
    print(f"📚 Subject: {subject}")
    print(f"🎯 Target date: {exam_date_str}")
    print(f"⏰ Daily hours: {daily_hours}")

    # Step 1: Calculate available study time
    # Parse dates from strings
    if today_str is None:
        # datetime.now(): Get current date and time
        today = datetime.now()
    else:
        # datetime.strptime(): Convert string to datetime object
        # '%Y-%m-%d': Date format (Year-Month-Day)
        today = datetime.strptime(today_str, '%Y-%m-%d')

    # Convert exam date string to datetime object
    exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d')

    # Calculate days between today and exam date
    # .days: Get only the days part of timedelta
    days_left = (exam_date - today).days

    # Check if exam date is in the future
    if days_left <= 0:
        return {
            "error": "❌ Exam date must be in the future!",
            "days_left": days_left,
            "exam_date": exam_date_str
        }

    # Calculate total available study hours
    # days_left * daily_hours: Total hours available for study
    total_hours = days_left * daily_hours

    print(f"📊 Days available: {days_left}")
    print(f"📊 Total study hours: {total_hours}")

    # Step 2: Get AI-powered course recommendations
    # Call our AI recommendation function
    # Use config for number of recommendations for planning
    recommendations = find_best_courses(
        user_topics, subject,
        top_n=RECOMMENDATION_CONFIG['max_recommendations_for_planning']
    )

    # Check if we got any recommendations
    if recommendations.empty:
        return {
            "error": "❌ No courses found matching your interests",
            "user_topics": user_topics,
            "subject": subject,
            "days_left": days_left,
            "total_hours": total_hours
        }

    # Step 3: Select courses that fit in available time
    selected_courses = []  # List to store selected courses
    used_hours = 0  # Track how many hours we've used

    # Loop through recommended courses
    # .iterrows(): Iterate over DataFrame rows
    for _, course in recommendations.iterrows():

        # Get course duration
        course_duration = course['content_duration']

        # Check if course fits in remaining time
        if used_hours + course_duration <= total_hours:
            # Add course to selected list
            selected_courses.append({
                'title': course['course_title'],
                'duration': course_duration,
                'level': course['level'],
                'similarity': course['similarity_score'],
                'students': course['num_subscribers'],
                'reviews': course['num_reviews'],
                'price': course.get('price_usd', 0)  # .get(): Get value or default
            })

            # Update used hours
            used_hours += course_duration

        else:
            # Course doesn't fit, check if we can do partial
            remaining_hours = total_hours - used_hours

            # If we have some remaining time, suggest partial completion
            if remaining_hours > 0:
                selected_courses.append({
                    'title': course['course_title'] + " (Partial)",
                    'duration': remaining_hours,
                    'level': course['level'],
                    'similarity': course['similarity_score'],
                    'students': course['num_subscribers'],
                    'reviews': course['num_reviews'],
                    'price': course.get('price_usd', 0),
                    'is_partial': True
                })

                # Use all remaining hours
                used_hours = total_hours

            # Stop adding courses (no more time)
            break

    # Step 4: Calculate schedule efficiency
    # What percentage of available time will be used
    if total_hours > 0:
        efficiency = (used_hours / total_hours) * 100
    else:
        efficiency = 0

    # Step 5: Calculate completion date
    # How many days will it actually take to complete selected courses
    if daily_hours > 0:
        actual_days_needed = used_hours / daily_hours
        # timedelta(): Create time difference
        completion_date = today + timedelta(days=int(actual_days_needed))
    else:
        completion_date = today

    # Step 6: Create daily schedule breakdown
    daily_schedule = create_daily_schedule(selected_courses, days_left, daily_hours)

    # Step 7: Return complete study plan
    return {
        'user_topics': user_topics,
        'subject': subject,
        'exam_date': exam_date_str,
        'days_left': days_left,
        'daily_hours': daily_hours,
        'total_hours': total_hours,
        'used_hours': used_hours,
        'selected_courses': selected_courses,
        'schedule_efficiency': efficiency,
        'completion_date': completion_date.strftime('%Y-%m-%d'),
        'daily_schedule': daily_schedule,
        'recommendations_count': len(recommendations)
    }

def create_daily_schedule(selected_courses, days_left, daily_hours):
    """
    Create a day-by-day schedule breakdown.

    Args:
        selected_courses (list): List of selected courses
        days_left (int): Number of days available
        daily_hours (int): Hours per day

    Returns:
        list: Daily schedule with courses and hours
    """

    # If no courses selected, return empty schedule
    if not selected_courses:
        return []

    # Calculate total duration of all courses
    total_duration = sum(course['duration'] for course in selected_courses)

    # If no duration, return empty schedule
    if total_duration == 0:
        return []

    daily_schedule = []  # List to store daily schedule
    current_day = 1  # Start with day 1
    remaining_daily_hours = daily_hours  # Hours left in current day
    course_index = 0  # Track which course we're on
    remaining_course_hours = selected_courses[0]['duration']  # Hours left in current course

    # Loop until we've scheduled all courses or run out of days
    while course_index < len(selected_courses) and current_day <= days_left:

        # Calculate how many hours to study today
        hours_today = min(remaining_daily_hours, remaining_course_hours)

        # Add to daily schedule
        daily_schedule.append({
            'day': current_day,
            'course': selected_courses[course_index]['title'],
            'hours': hours_today,
            'level': selected_courses[course_index]['level'],
            'progress': f"{hours_today}/{selected_courses[course_index]['duration']} hours"
        })

        # Update remaining hours
        remaining_daily_hours -= hours_today
        remaining_course_hours -= hours_today

        # Check if course is completed
        if remaining_course_hours <= 0:
            # Move to next course
            course_index += 1
            if course_index < len(selected_courses):
                remaining_course_hours = selected_courses[course_index]['duration']

        # Check if day is completed
        if remaining_daily_hours <= 0:
            # Move to next day
            current_day += 1
            remaining_daily_hours = daily_hours

    return daily_schedule

def display_study_plan(plan):
    """
    Display study plan in a user-friendly format.

    Args:
        plan (dict): Study plan from create_study_plan()
    """

    # Check if there was an error
    if 'error' in plan:
        print(f"{plan['error']}")
        return

    # Display header
    print("\n" + "="*80)
    print("📅 AI-POWERED PERSONALIZED STUDY PLAN")
    print("="*80)

    # Display overview
    print(f"🎯 Your Learning Goals: {', '.join(plan['user_topics'])}")
    print(f"📚 Subject Area: {plan['subject']}")
    print(f"📅 Target Exam Date: {plan['exam_date']}")
    print(f"⏰ Days Available: {plan['days_left']}")
    print(f"📊 Study Hours Per Day: {plan['daily_hours']}")
    print(f"⏱️  Total Available Hours: {plan['total_hours']}")
    print(f"✅ Planned Study Hours: {plan['used_hours']:.1f}")
    print(f"🎯 Schedule Efficiency: {plan['schedule_efficiency']:.1f}%")
    print(f"🏁 Estimated Completion: {plan['completion_date']}")
    print()

    # Display selected courses
    print("📋 RECOMMENDED COURSES:")
    print("-" * 60)

    if plan['selected_courses']:
        # Loop through selected courses
        # enumerate(): Add numbers starting from 1
        for i, course in enumerate(plan['selected_courses'], 1):
            print(f"{i}. {course['title']}")
            print(f"   📊 Relevance: {course['similarity']*100:.1f}%")
            print(f"   ⏱️  Duration: {course['duration']:.1f} hours")
            print(f"   📈 Level: {course['level']}")
            print(f"   👥 Students: {course['students']:,}")
            print(f"   ⭐ Reviews: {course['reviews']:,}")

            # Check if it's a partial course
            if course.get('is_partial', False):
                print(f"   ⚠️  Note: Partial completion due to time constraints")

            print()

    else:
        print("❌ No courses selected (not enough time)")

    # Display daily schedule
    if plan['daily_schedule']:
        print("\n📅 DAILY SCHEDULE:")
        print("-" * 60)

        current_day = 0
        for schedule_item in plan['daily_schedule']:
            if schedule_item['day'] != current_day:
                current_day = schedule_item['day']
                print(f"\n📅 Day {current_day}:")

            print(f"   📚 {schedule_item['course']}")
            print(f"   ⏱️  Study: {schedule_item['hours']:.1f} hours")
            print(f"   📈 Level: {schedule_item['level']}")
            print(f"   📊 Progress: {schedule_item['progress']}")

    print("\n" + "="*80)

def get_study_recommendations(subject, available_hours):
    """
    Get general study recommendations based on available time.

    Args:
        subject (str): Subject area
        available_hours (int): Total hours available for study

    Returns:
        dict: Study recommendations and tips
    """

    recommendations = {
        'subject': subject,
        'available_hours': available_hours,
        'tips': []
    }

    # General study tips based on available time
    if available_hours < 10:
        recommendations['tips'].extend([
            "⚡ Focus on fundamentals and key concepts",
            "📝 Use active learning techniques (flashcards, practice problems)",
            "🎯 Prioritize high-impact topics",
            "⏰ Use time-blocking for focused study sessions"
        ])

    elif available_hours < 50:
        recommendations['tips'].extend([
            "📚 Balance theory and practical application",
            "🔄 Use spaced repetition for better retention",
            "💡 Focus on understanding concepts, not just memorization",
            "🎯 Practice with real-world examples"
        ])

    else:
        recommendations['tips'].extend([
            "🏗️ Build a strong foundation before advanced topics",
            "🔍 Dive deep into complex concepts",
            "💼 Work on practical projects",
            "🎓 Prepare for advanced certifications"
        ])

    # Subject-specific tips
    if 'finance' in subject.lower() or 'business' in subject.lower():
        recommendations['tips'].extend([
            "📊 Practice with Excel and financial modeling",
            "📈 Study real market cases and examples",
            "💰 Learn both theory and practical applications"
        ])

    elif 'web' in subject.lower() or 'development' in subject.lower():
        recommendations['tips'].extend([
            "💻 Build projects while learning",
            "🔧 Practice coding daily",
            "🌐 Stay updated with latest technologies"
        ])

    return recommendations

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_study_planner():
    """
    Test the study planning system with sample data.
    """

    print("🧪 Testing AI Study Planning System")
    print("="*50)

    # Test Case 1: Normal study plan
    print("\n📋 Test Case 1: Business Finance Study Plan")
    topics = ["excel", "financial analysis", "investment"]
    subject = "Business Finance"
    exam_date = "2025-03-15"
    daily_hours = 3

    plan = create_study_plan(topics, subject, exam_date, daily_hours)
    display_study_plan(plan)

    # Test Case 2: Short timeline
    print("\n📋 Test Case 2: Short Timeline Study Plan")
    topics2 = ["javascript", "html", "css"]
    subject2 = "Web Development"
    exam_date2 = "2025-02-01"
    daily_hours2 = 2

    plan2 = create_study_plan(topics2, subject2, exam_date2, daily_hours2)
    display_study_plan(plan2)

    # Test Case 3: Study recommendations
    print("\n📋 Test Case 3: Study Recommendations")
    recs = get_study_recommendations("Business Finance", 30)
    print(f"📚 Recommendations for {recs['subject']} ({recs['available_hours']} hours):")
    for tip in recs['tips']:
        print(f"   {tip}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# if __name__ == "__main__": Only run this code when file is executed directly
if __name__ == "__main__":

    # Run study planner tests
    test_study_planner()
