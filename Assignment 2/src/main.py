__author__ = "Dario Mameli"

import sys

from ContentBasedRecommender import *


if __name__ == '__main__':

    # Build the recommender
    CBRecommender = ContentBasedRecommender("dataset/movies.csv", "dataset/ratings.csv")

    # Question 1
    print("QUESTION 1")
    genres = CBRecommender.get_genres()
    genres_frequencies = []
    print("Genre\tFrequency")
    for genre in genres:
        freq = CBRecommender.compute_genre_frequency(genre)
        print(f"{genre}\t{freq}")
        genres_frequencies.append(freq)
    print()

    # Question 2
    print(f"QUESTION 2 \nNum of genres = {len(genres)}\n")

    # Question 3
    print("QUESTION 3")
    indices = np.argsort(genres_frequencies)
    most_common = [genres[i] for i in indices[-5:]]
    least_common = [genres[i] for i in indices[:5]]
    print(f"Five most common genres: {most_common}\n"
          f"Five least common genres: {least_common}")
    print()

    # Question 4
    print("QUESTION 4")
    min_, max_, avg = CBRecommender.compute_min_max_average_genres()
    print(f"Min: {min_}, Max: {max_}, Average: {avg}")
    print()

    # Question 5
    print("QUESTION 5")
    userID = 289
    movieID = 1125
    CBRecommender.retrieve_rating_info(userID, movieID)
    print()

    # Question 6
    print("QUESTION 6")
    userID = 526
    CBRecommender.retrieve_rating_history(userID)
    print()
    CBRecommender.compute_genres_frequencies(userID)
    print()
    CBRecommender.study_ratings(userID)
    print()
    CBRecommender.get_user_profile(userID)
    print()

    # Question 7
    print("QUESTION 7")
    CBRecommender.topN(userID, 5)
    print()

    # Question 8
    print("QUESTION 8")
    CBRecommender.topN(userID, 5, movies_normalization=True)
    print()

    # Question 9
    print("QUESTION 9")
    userID = 14
    CBRecommender.retrieve_rating_history(userID)
    print()
    CBRecommender.get_user_profile(userID)
    print()
    CBRecommender.study_ratings(userID)
    print()

    # Question 10
    print("QUESTION 10")
    userID = 14
    CBRecommender.topN(userID, 5, movies_normalization=True)
    print()

    # Question 11
    print("QUESTION 11")
    idf = CBRecommender.IDF()
    print("Genre\tIDF")
    for k in idf:
        print(f"{k}\t{idf[k]:.6f}")
    print()

    # Question 12
    print("QUESTION 12")
    userID = 526
    CBRecommender.topN(userID, 5, movies_normalization=True, users_rescaling=True)
    print()

    # Question 13
    print("QUESTION 13")
    userID = 526
    CBRecommender.topN_rerank(userID, n=5, n_in=200, lambda_=0.5, movies_normalization=True, users_rescaling=True)
    print()

    # Question 14
    print("QUESTION 14")
    userID = 225
    CBRecommender.retrieve_rating_history(userID)
    CBRecommender.compute_genres_frequencies(userID)
    CBRecommender.study_ratings(userID)
    CBRecommender.study_ratings_per_genre(userID)
    CBRecommender.get_user_profile(userID)
    CBRecommender.topN_rerank(userID, n=5, n_in=200, lambda_=0.5, movies_normalization=True, users_rescaling=True)
    print()

    # Question 15
    print("QUESTION 15")
    userID = 341
    CBRecommender.retrieve_rating_history(userID)
    CBRecommender.compute_genres_frequencies(userID)
    CBRecommender.study_ratings(userID)
    CBRecommender.study_ratings_per_genre(userID)
    CBRecommender.get_user_profile(userID, movies_normalization=True, users_rescaling=True)
    CBRecommender.topN_rerank(userID, n=5, n_in=200, lambda_=0.5, movies_normalization=True, users_rescaling=True)
    print()




