from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the CSV data into a DataFrame
data = pd.read_csv('data4.csv')

# Function to recommend places based on city and preferences
def recommend_places(city, preferences):
    # Filter data based on city
    filtered_data = data[data['city'] == city]

    # Filter data based on preferences
    recommendations = pd.DataFrame(columns=data.columns)
    for pref in preferences:
        recommendations = pd.concat([recommendations, filtered_data[filtered_data['category'].str.contains(pref.strip(), case=False)]])
    
    # Sort recommendations by rating and count
    recommendations = recommendations.sort_values(by=['p_rating', 'count'], ascending=False)
    
    return recommendations[['title', 'category', 'p_rating', 'count']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    city = request.form['city'].strip().lower()
    preferences = request.form['preferences'].strip().split(',')
    
    # Get recommendations
    recommendations = recommend_places(city, preferences)

    # Render the recommendations template with the recommendations data
    return render_template('recommendations.html', recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
