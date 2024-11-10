from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.spatial.distance import correlation

app = Flask(__name__)

data = pd.read_csv('data4_1.csv')
placeInfo = pd.read_csv('data4.csv')

data = pd.merge(data, placeInfo, left_on='itemId', right_on="itemId")

userItemRatingMatrix = None  # Define it as None initially

def similarity(user1, user2):
    try:
        user1 = np.array(user1) - np.nanmean(user1)
        user2 = np.array(user2) - np.nanmean(user2)
        commonItemIds = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]
        if len(commonItemIds) == 0:
            return 0
        else:
            user1 = np.array([user1[i] for i in commonItemIds])
            user2 = np.array([user2[i] for i in commonItemIds])
            return correlation(user1, user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")

def nearestNeighbourRatings(activeUser, K):
    global userItemRatingMatrix  # Declare it as global
    try:
        userItemRatingMatrix = pd.pivot_table(data, values='rating', index=['userId'], columns=['itemId'])
        similarityMatrix = pd.DataFrame(index=userItemRatingMatrix.index, columns=['Similarity'])
        for i in userItemRatingMatrix.index:
            similarityMatrix.loc[i] = similarity(userItemRatingMatrix.loc[activeUser], userItemRatingMatrix.loc[i])
        similarityMatrix = pd.DataFrame.sort_values(similarityMatrix, ['Similarity'], ascending=[0])
        nearestNeighbours = similarityMatrix[:K]
        neighbourItemRatings = userItemRatingMatrix.loc[nearestNeighbours.index]
        predictItemRating = pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        for i in userItemRatingMatrix.columns:
            predictedRating = np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j, i] > 0:
                    predictedRating += (userItemRatingMatrix.loc[j, i] - np.nanmean(userItemRatingMatrix.loc[j])) * nearestNeighbours.loc[j, 'Similarity']
            predictItemRating.loc[i, 'Rating'] = predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return predictItemRating

def topNRecommendations(activeUser, N):
    try:
        predictItemRating = nearestNeighbourRatings(activeUser, 10)
        placeAlreadyWatched = list(userItemRatingMatrix.loc[activeUser].loc[userItemRatingMatrix.loc[activeUser] > 0].index)
        predictItemRating = predictItemRating.drop(placeAlreadyWatched)
        topRecommendations = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[:N]
        topRecommendationTitles = (placeInfo.loc[placeInfo.itemId.isin(topRecommendations.index)])
        recommendations = list(topRecommendationTitles.title)
        return recommendations
    except ZeroDivisionError:
        print("You can't divide by zero!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    activeUser = int(request.form['userid'])
    recommendations = topNRecommendations(activeUser, 4)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
