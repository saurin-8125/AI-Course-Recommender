# 🎓 AI Course Recommender

An intelligent course recommendation system that uses TF-IDF and cosine similarity to find the perfect courses for your learning journey and creates personalized study plans.

## 🚀 Features

- **AI-Powered Recommendations**: Uses TF-IDF vectorization and cosine similarity to find courses matching your interests
- **Personalized Study Plans**: Creates detailed study schedules based on your available time and target dates
- **Interactive Web Interface**: Beautiful Streamlit web app for easy interaction
- **Detailed Analytics**: Comprehensive dataset analysis and course statistics
- **Multiple Subjects**: Supports Business Finance, Web Development, Graphic Design, and Musical Instruments

## 🧠 How It Works

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- Converts course descriptions into numerical vectors
- Identifies important and unique words in each course
- Focuses on words that are both frequent in a course AND rare across all courses

### 2. Cosine Similarity
- Measures similarity between your interests and course content
- Returns similarity scores from 0% (no match) to 100% (perfect match)
- Ranks courses by relevance to your learning goals

### 3. Study Planning Algorithm
- Analyzes your available time and target dates
- Selects optimal courses that fit your schedule
- Creates day-by-day study plans with progress tracking

## 📁 Project Structure

```
AI_Course_Recommender/
├── 📄 README.md                     # This file
├── 📄 requirement.txt               # Python dependencies
├── 📄 .gitignore                    # Git ignore file
├── 📂 data/
│   └── 📄 courses.csv              # Course dataset (3,678 courses)
├── 📂 app/
│   ├── 📄 __init__.py              # Package initialization
│   ├── 📄 topic_matcher.py         # AI recommendation engine
│   ├── 📄 data_loader.py           # Data loading and cleaning
│   └── 📄 study_planner.py         # Study plan creation
├── 📂 ui/
│   └── 📄 streamlit_app.py         # Web interface
└── 📂 learning/
    ├── 📄 simple_recommender.py    # Learning examples
    └── 📄 explain_everything.py    # Detailed explanations
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd AI_Course_Recommender
```

### Step 2: Install Dependencies
```bash
pip install -r requirement.txt
```

### Step 3: Verify Installation
```bash
cd app
python topic_matcher.py
```

You should see output like:
```
✅ Successfully loaded 3678 courses from ../data/courses.csv
📚 Available subjects: ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development']
```

## 🚀 Usage

### Option 1: Web Interface (Recommended)
```bash
cd ui
streamlit run streamlit_app.py
```

This opens a beautiful web interface where you can:
- Select your subject area
- Enter topics you're interested in
- Set your target dates and daily study hours
- Get AI-powered course recommendations
- Create personalized study plans

### Option 2: Command Line Interface
```python
from app.topic_matcher import find_best_courses
from app.study_planner import create_study_plan

# Get course recommendations
topics = ["excel", "financial analysis", "investment"]
subject = "Business Finance"
recommendations = find_best_courses(topics, subject, top_n=5)
print(recommendations)

# Create study plan
plan = create_study_plan(
    user_topics=topics,
    subject=subject,
    exam_date_str="2025-03-15",
    daily_hours=2
)
```

## 📊 Dataset

The system uses a comprehensive dataset of 3,678 courses with the following features:

- **course_title**: Name of the course
- **subject**: Subject category (Business Finance, Web Development, etc.)
- **content_duration**: Course length in hours
- **level**: Difficulty level (Beginner, Intermediate, Advanced, Expert)
- **num_subscribers**: Number of students enrolled
- **num_reviews**: Number of course reviews
- **price_usd**: Course price in USD
- **text_for_recommendations**: Combined text for AI analysis

## 🔍 Code Examples

### Basic Course Recommendation
```python
from app.topic_matcher import find_best_courses, display_recommendations

# Define your interests
my_interests = ["python", "data analysis", "machine learning"]
my_subject = "Business Finance"

# Get recommendations
results = find_best_courses(my_interests, my_subject, top_n=3)

# Display results
display_recommendations(results, my_interests)
```

### Study Plan Creation
```python
from app.study_planner import create_study_plan, display_study_plan

# Create study plan
plan = create_study_plan(
    user_topics=["javascript", "html", "css"],
    subject="Web Development",
    exam_date_str="2025-04-01",
    daily_hours=3
)

# Display plan
display_study_plan(plan)
```

### Data Analysis
```python
from app.data_loader import load_courses, display_data_info

# Load and analyze data
df = load_courses()
display_data_info(df)
```

## 🧠 Understanding the AI

### TF-IDF Vectorization
```python
# Example of how TF-IDF works
course_text = "Python programming for data analysis"
user_query = "python data"

# TF-IDF converts these to vectors like:
# course_vector = [0.8, 0.0, 0.7, 0.9, 0.0, ...]
# user_vector =   [0.8, 0.0, 0.0, 0.9, 0.0, ...]

# Where each number represents word importance
```

### Cosine Similarity
```python
# Similarity calculation example
# cosine_similarity(user_vector, course_vector) = 0.85
# This means 85% similarity between user interests and course content
```

## 📈 Performance Metrics

- **Accuracy**: Relevance scores typically range from 60-90% for good matches
- **Speed**: Processes 3,678 courses in under 2 seconds
- **Memory**: Uses sparse matrices for efficient storage
- **Coverage**: Supports 4 major subject areas with 3,678+ courses

## 🎯 Key Functions Explained

### topic_matcher.py
- `load_course_data()`: Loads and validates course data
- `find_best_courses()`: Core AI recommendation function
- `display_recommendations()`: Formats and displays results

### study_planner.py
- `create_study_plan()`: Creates personalized study schedules
- `create_daily_schedule()`: Breaks down courses into daily tasks
- `display_study_plan()`: Shows formatted study plans

### data_loader.py
- `load_courses()`: Basic data loading with error handling
- `clean_data()`: Prepares data for AI processing
- `display_data_info()`: Shows dataset statistics

## 🔧 Customization

### Adding New Subjects
1. Add new course data to `data/courses.csv`
2. Ensure the `subject` column includes your new category
3. The system will automatically recognize new subjects

### Modifying AI Parameters
```python
# In topic_matcher.py, modify TfidfVectorizer parameters
vectorizer = TfidfVectorizer(
    stop_words='english',     # Language for stop words
    max_features=1000,        # Number of features to keep
    ngram_range=(1, 2)        # Single words and bigrams
)
```

### Adjusting Study Plans
```python
# In study_planner.py, modify planning logic
# Change how courses are selected or scheduled
```

## 📚 Learning Resources

### For Beginners
- Check `learning/simple_recommender.py` for basic examples
- Read `learning/explain_everything.py` for detailed explanations
- Start with the web interface for easier interaction

### For Advanced Users
- Modify AI parameters in `topic_matcher.py`
- Customize study planning algorithms
- Add new features to the Streamlit interface

## 🐛 Troubleshooting

### Common Issues

1. **"Could not find file" error**
   - Ensure you're running from the correct directory
   - Check that `data/courses.csv` exists

2. **"No recommendations found"**
   - Try different topic keywords
   - Check available subjects with `get_available_subjects()`

3. **Streamlit not starting**
   - Ensure streamlit is installed: `pip install streamlit`
   - Check port availability (default: 8501)

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 Dependencies

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms (TF-IDF, cosine similarity)
- **streamlit**: Web interface framework
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📊 Project Statistics

- **Total Courses**: 3,678
- **Subjects**: 4 major areas
- **Code Files**: 7 main modules
- **Functions**: 25+ documented functions
- **Lines of Code**: 1,500+ (heavily commented)

## 🎓 Educational Value

This project demonstrates:
- **Machine Learning**: TF-IDF vectorization and cosine similarity
- **Data Science**: Data cleaning, analysis, and visualization
- **Web Development**: Interactive Streamlit applications
- **Software Engineering**: Modular design and documentation
- **AI Applications**: Practical recommendation systems

## 📞 Support

For questions or issues:
1. Check the `learning/` directory for examples
2. Review function docstrings for detailed explanations
3. Run individual modules to test functionality
4. Check the troubleshooting section above

## 🏆 Acknowledgments

- Dataset sourced from Udemy course catalog
- Built with Python's scientific computing stack
- Inspired by modern recommendation systems (Netflix, Amazon, etc.)

---

**Happy Learning! 🎓**

*Find the perfect courses for your learning journey with AI-powered recommendations!*
