__author__ = "Dario Mameli"

import sys
from HybridRecommender import *


if __name__ == '__main__':

    # Initial setup
    path_movies = "dataset/movies.csv"
    path_ratings = "dataset/ratings.csv"

    # ------------------------------------------ ANALYSIS --------------------------------------------------------------

    print("-" * 50, "ANALYSIS", "-" * 50)
    print()

    print("-" * 50, "UUCF", "-" * 50)
    uucf = UUCFRecommender(path_movies, path_ratings)

    # QUESTION 1
    print("QUESTION 1")
    p_corr_1 = uucf.pearson_correlation(1, 4, gamma=None)
    print(f"Pearson correlation between user 1 and 4 without significance weighting"
          f"\n\t{p_corr_1}")
    print()

    # QUESTION 2
    print("QUESTION 2")
    p_corr_2 = uucf.pearson_correlation(1, 4)
    print(f"Pearson correlation between user 1 and 4 with significance weighting"
          f"\n\t{p_corr_2}")
    print()

    # QUESTION 3 & 4
    print("QUESTION 3 & 4")
    ratio = p_corr_1/p_corr_2
    print(f"Ratio: {ratio}")
    print()

    # QUESTION 5
    userID = 1
    itemID = 10
    print("QUESTION 5")
    print(f"Number of neighbors of user {userID} item {itemID}: {len(uucf.neighbors(userID, itemID))}")
    print()

    # QUESTION 6
    print("QUESTION 6")
    print(f"Neighbors of user {userID} item {itemID}")
    topK = uucf.topK_neighbors(userID, itemID)
    printTopKNeighborsUser(topK)
    print()

    # QUESTION 7
    print("QUESTION 7")
    print("\t", uucf.weighted_average_deviation_meanRating(userID, itemID))
    print()

    # QUESTION 8
    print("QUESTION 8")
    print(f"Rating prediction user {userID} item {itemID}:")
    print("\t", uucf.predict_score(userID, itemID))
    print()

    # QUESTION 9
    print("QUESTION 9")
    print(f"Title of movie {itemID}:")
    print("\t", uucf.get_title(itemID))
    print()

    # QUESTION 10
    userID = 1
    itemID = 260
    print("QUESTION 10")
    print(f"Number of neighbors of user {userID} item {itemID}: {len(uucf.neighbors(userID, itemID))}")
    print()

    # QUESTION 11
    print("QUESTION 11")
    print(f"Neighbors of user {userID} item {itemID}")
    topK = uucf.topK_neighbors(userID, itemID)
    printTopKNeighborsUser(topK)
    print()

    # QUESTION 12
    print("QUESTION 12")
    print(f"Neighbors of user {userID} item {itemID}")
    print("\t", uucf.weighted_average_deviation_meanRating(userID, itemID))
    print()

    # QUESTION 13
    print("QUESTION 13")
    print(f"Rating prediction user {userID} item {itemID}:")
    print("\t", uucf.predict_score(userID, itemID))
    print()

    # QUESTION 14
    print("QUESTION 14")
    print(f"Title of movie {itemID}:")
    print("\t", uucf.get_title(itemID))
    print()

    # QUESTION 16
    print("QUESTION 16")
    uucf.topN(1)
    print()

    # QUESTION 17
    print("QUESTION 17")
    userID = 522
    uucf.retrieve_rating_history(userID)
    print()
    topN = uucf.topN(userID)
    print()
    for m in topN:
        print(f"Movie {m[0]}")
        neighbors = uucf.topK_neighbors(userID, m[0], k=5)
        for n in neighbors:
            uucf.retrieve_rating_history(n[0])
            print()
        print()
    print()

# ------------------------------------------------------------------

    print("-" * 50, "IICF", "-" * 50)
    iicf = IICFRecommender(path_movies, path_ratings)

    # QUESTION 19
    print("QUESTION 19")
    print(f"Strict positive similarities:"
          f"\n\t{iicf.count_positive_elements_model()}")
    print()

    # QUESTION 20
    print("QUESTION 20")
    item1 = 594
    item2 = 596
    print(f"Cosine similarity item {item1} item {item2}")
    print(f"\tMovie {item1}: {iicf.get_title(item1)}")
    print(f"\tMovie {item2}: {iicf.get_title(item2)}")
    print(f"\t{iicf.cosine_similarity(item1, item2)}")
    print()

    # QUESTION 21
    print("QUESTION 21")
    userID = 522
    itemID = 25
    neighbors = iicf.neighbors(userID, itemID)
    print(f"Number of neighbors user {userID}, item {itemID}"
          f"\n\t{len(neighbors)}")
    print()

    # QUESTION 22
    print("QUESTION 22")
    userID = 522
    itemID = 25
    topK = neighbors[:20]
    print(f"Top 20 most similar items user {userID}, item {itemID}")
    printTopKNeighborsMovie(topK)
    print()

    # QUESTION 24
    print("QUESTION 24")
    userID = 522
    print(f"Top 10 recommendations user {userID}")
    iicf.topN(userID)
    print()

# --------------------------------------------------------------

    print("-" * 50, "Basket recommendations", "-" * 50)

    # QUESTION 25
    print("QUESTION 25")
    basket = [1]
    print("Basket")
    for movie_id in basket:
        print(f"Movie {movie_id}: {iicf.get_title(movie_id)}")
    print(f"Top 10 basket {basket}")
    iicf.topN_basket(basket)
    print()

    # QUESTION 26
    print("QUESTION 26")
    basket = [1, 48, 239]
    print("Basket")
    for movie_id in basket:
        print(f"Movie {movie_id}: {iicf.get_title(movie_id)}")
    print(f"Top 10 basket {basket}")
    iicf.topN_basket(basket)
    print()

    # QUESTION 27
    print("QUESTION 27")
    basket = [1, 48, 239]
    print("Basket")
    for movie_id in basket:
        print(f"Movie {movie_id}: {iicf.get_title(movie_id)}")
    print(f"Top 10 basket {basket}")
    iicf.topN_basket(basket, all_sim=True)
    print()

# --------------------------------------------------------------

    print("-" * 50, "Hybrid", "-" * 50)

    hybrid = HybridRecommender(uucf, iicf)

    # QUESTION 29
    print("QUESTION 29")
    userID = 522
    print(f"Top 10 recommendations user {userID}")
    hybrid.topN(userID)
    print()
