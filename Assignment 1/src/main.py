__author__ = "Dario Mameli"

import sys

from methods import *


if __name__ == '__main__':

    # Load datasets
    datasetMovies = read_dataset("dataset/movies.dat")
    datasetUsers = read_dataset("dataset/users.dat")
    datasetRatings = read_dataset("dataset/ratings.dat")

#---------------------------------Analysis ---------------------------------------------------------------------------

    # QUESTION 1, 2, 4, 7
    sim_ass_1_1064 = simple_association(datasetRatings, 1, 1064)
    print("Sim. Ass. 1-1064:", sim_ass_1_1064)
    adv_ass_1_1064 = advanced_association(datasetRatings, 1, 1064)
    print("Adv. Ass. 1-1064:", adv_ass_1_1064)
    sim_ass_1_2858 = simple_association(datasetRatings, 1, 2858)
    print("Sim. Ass. 1-2858:", sim_ass_1_2858)
    adv_ass_1_2858 = advanced_association(datasetRatings, 1, 2858)
    print("Adv. Ass. 1-2858:", adv_ass_1_2858)

    print()

    # QUESTION 5
    # Define the movie IDs
    movie_ids = [1, 1064, 2858]
    # Filter datasetMovies to only consider the rows with the specified MovieIDs
    filtered_movies_dataset = datasetMovies[datasetMovies['MovieID'].isin(movie_ids)]
    # Print the titles and genres and then the frequencies of each movie
    for index, row in filtered_movies_dataset.iterrows():
        movie_id = row['MovieID']
        # Calculate the frequency of the current MovieID in datasetRatings
        frequency = calculate_frequency(datasetRatings, movie_id)
        print(f"MovieID: {row['MovieID']}, Title: {row['Title']}, Genres: {row['Genres']}")
        print("Frequency:", frequency)

    print()

    # QUESTION 8
    max_sim_ass = max(sim_ass_1_1064, sim_ass_1_2858)
    if max_sim_ass == sim_ass_1_1064:
        print("Max sim. ass. is 1-1064")
    else:
        print("Max sim. ass. is 1-2858")
    max_adv_ass = max(adv_ass_1_1064, adv_ass_1_2858)
    if max_adv_ass == adv_ass_1_1064:
        print("Max adv. ass. is 1-1064")
    else:
        print("Max adv. ass. is 1-2858")

    print()

    # QUESTION 9, 14
    top10_freq = topN_frequency(datasetRatings, 10)
    print("Top 10 - All ratings")
    for i, element in enumerate(top10_freq):
        movieID = element[0]
        title = findTitle(datasetMovies, movieID)
        num_users = findNumUsers(datasetRatings, movieID)
        user_movie_counts = datasetRatings.groupby(["UserID", "MovieID"]).size()
        print(i+1, "MovieID:", movieID, "Num. Users:", element[1], "Movie title:", title)
    top10_freq = topN_frequency(datasetRatings, 10, 4)
    print("Top 10 - >=4 stars ratings")
    for i, element in enumerate(top10_freq):
        movieID = element[0]
        title = findTitle(datasetMovies, element[0])
        num_users = findNumUsers(datasetRatings, movieID)
        user_movie_counts = datasetRatings.groupby(["UserID", "MovieID"]).size()
        print(i + 1, "MovieID:", movieID, "Num. Users:", element[1], "Movie title:", title)

    print()

    # QUESTION 10, 11
    queryMovieID = 3941
    print("MovieID:", queryMovieID, "Title:", findTitle(datasetMovies, queryMovieID))
    print("Frequency:", calculate_frequency(datasetRatings, queryMovieID))
    print("Top 5 - Simple associations:")
    top5_sim_3941 = topN_simple(datasetRatings, queryMovieID, 5)
    for rank in top5_sim_3941:
        movieID = top5_sim_3941[rank][0]
        ass_value = top5_sim_3941[rank][1]
        title = findTitle(datasetMovies, movieID)
        print(rank, "MovieID:", movieID, "Ass. Value", ass_value, "Movie title:", title)
        print("Frequency:", calculate_frequency(datasetRatings, movieID))
    print("Top 5 - Advanced associations:")
    top5_adv_3941 = topN_advanced(datasetRatings, queryMovieID, 5)
    for rank in top5_adv_3941:
        movieID = top5_adv_3941[rank][0]
        ass_value = top5_adv_3941[rank][1]
        title = findTitle(datasetMovies, movieID)
        print(rank, "MovieID:", movieID, "Ass. Value", ass_value, "Movie title:", title)
        print("Frequency:", calculate_frequency(datasetRatings, movieID))
