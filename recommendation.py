import pickle

# Load models and data
with open('cv_model.pkl', 'rb') as file:
    loaded_cv = pickle.load(file)

with open('similarity_matrix.pkl', 'rb') as file:
    loaded_similarity = pickle.load(file)

with open('courses_data.pkl', 'rb') as file:
    loaded_df = pickle.load(file)

# Recommendation function
def recommend(course):
    course = course.lower()
    course_names = loaded_df['course_name'].str.lower()

    if course not in course_names.values:
        suggestions = course_names[course_names.str.contains(course.split()[0])]
        return suggestions.head(5).tolist()

    course_index = course_names[course_names == course].index[0]
    distances = loaded_similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]

    recommendations = []
    for i in course_list:
        recommendations.append(loaded_df.iloc[i[0]].course_name)

    return recommendations


