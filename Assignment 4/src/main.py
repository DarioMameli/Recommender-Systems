__author__ = "Dario Mameli"

from LensKitRecommender import *
import sys
import os


if __name__ == '__main__':

    # Build the recommender
    lkrec = LensKitRecommender('dataset/ml-100k')

    # Set user and item ids
    userID = 196
    itemID = 302

# ----------------------------- ANALYSIS -----------------------------

    # QUESTION 6
    print("QUESTION 6")
    for i in range(2):
        print("-" * 50, f"ITERATION {i+1}", "-" * 50)
        lkrec.evaluate_recommendations()
        print()

    # QUESTION 8
    print("QUESTION 8")
    print("Predicting scores..")
    for alg in ALGORITHMS:
        print(lkrec.predict_for_user(userID, itemID, alg[0]))
        print()

    # QUESTION 10
    print("QUESTION 10")
    for i in range(3):
        print("-" * 50, f"Trial {i+1}", "-" * 50)
        print("Predict scores..")
        for alg in ALGORITHMS:
            print(lkrec.predict_for_user(userID, itemID, alg[0]))
            print()

    # QUESTION 11
    print("QUESTION 11")
    for i in range(5):
        print(lkrec.predict_for_user_withRandom(userID, itemID))
