__author__ = "Dario Mameli"


import pandas as pd
import numpy as np


ratings_headers = ["UserID", "MovieID", "Rating", "Timestamp"]
users_headers = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
movies_headers = ["MovieID", "Title", "Genres"]


def read_dataset(path):
    """
        Function that reads the dataset from a specified path and stores it in a DataFrame

        Parameters:
        path (string): The path to the dataset.

        Returns:
        DataFrame: The dataset in a suitable format for manipulation.
    """
    dataset = pd.read_csv(path, sep='::', header=None, engine='python', encoding='latin1')  # standard encoding utf-8
    # is not the correct one
    substrings = path.split("/")
    for token in substrings:  # check which dataset we are loading and assign the corresponding column names
        if token == "ratings.dat":
            dataset.columns = ratings_headers
        elif token == "users.dat":
            dataset.columns = users_headers
        elif token == "movies.dat":
            dataset.columns = movies_headers
    return dataset


def simple_association(dataset, movieX, movieY):
    """
        Function that computes the simple association between two movies

        Parameters:
        dataset (DataFrame): The dataset to manipulate.
        movieX (int): The first MovieID.
        movieY (int): The second MovieID.

        Returns:
        float: The simple association between movieX and movieY.
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    # Utility variables
    datasetX = dataset[dataset["MovieID"] == movieX]  # filter the dataset getting all entries with movieX
    datasetY = dataset[dataset["MovieID"] == movieY]  # filter the dataset getting all entries with movieY
    user_ids_X = set(datasetX["UserID"])  # build a set for the UserIDs related to movieX (no duplicated IDs)
    user_ids_Y = set(datasetY["UserID"])  # build a set for the UserIDs related to movieX (no duplicated IDs)
    user_ids_XY = user_ids_X.intersection(user_ids_Y)  # find common UserIDs from the previously defined sets

    # Variables to use for calculation
    X = (dataset["MovieID"] == movieX).sum()  # count occurrences of movieX
    X_and_Y = len(user_ids_XY)  # count co-occurrences of movieX and movieY

    # Simple association formula
    if X == 0:  # avoid dividing by 0
        return np.inf
    association = X_and_Y / X

    return association


def advanced_association(dataset, movieX, movieY):
    """
        Function that computes the advanced association between two movies

        Parameters:
        dataset (DataFrame): The dataset to manipulate.
        movieX (int): The first MovieID.
        movieY (int): The second MovieID.

        Returns:
        float: The advanced association between movieX and movieY.
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    # Utility variables (DataFrames and Sets)
    datasetX = dataset[dataset["MovieID"] == movieX]  # filter the dataset getting all entries with movieX
    datasetY = dataset[dataset["MovieID"] == movieY]  # filter the dataset getting all entries with movieY
    user_ids_X = set(datasetX["UserID"])  # build a set for the UserIDs related to movieX (no duplicated IDs)
    user_ids_Y = set(datasetY["UserID"])  # build a set for the UserIDs related to movieX (no duplicated IDs)
    user_ids = set(dataset["UserID"])  # build a set for all the UserIDs
    user_ids_XY = user_ids_X.intersection(user_ids_Y)  # find common UserIDs from the previously defined sets

    # Variables to use for calculation
    X = (dataset["MovieID"] == movieX).sum()  # count occurrences of movieX
    X_and_Y = len(user_ids_XY)  # count co-occurrences of movieX and movieY
    notX = len(user_ids) - len(user_ids_X)  # count number of users who did not rate movieX using a simple difference
    Y = (dataset["MovieID"] == movieY).sum()  # count occurrences of movieY
    notX_and_Y = Y - X_and_Y  # count number of users who did not rate movieX but rated movieY

    # Advanced association formula
    if X == 0:  # if X is not present then the association is non-existing
        return 0
    elif notX_and_Y == 0:  # avoid dividing by 0
        return np.inf  # maximum association as the two movies are only seen together in the data
    association = (X_and_Y / X) * (notX / notX_and_Y)

    return association


def topN_simple(dataset, movieID, n):
    """
        Function that provides the ranking of the top N MovieIDs based on simple association with the given MovieID

        Parameters:
        dataset (DataFrame): The dataset to manipulate.
        movieID (int): The query MovieID.
        n (int): The number of the best associations to find.

        Returns:
        dict: The ranking of the best n associations with movieID.
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    movieIDs = dataset["MovieID"].unique()  # get all MovieIDs
    movieIDs = [movie_id for movie_id in movieIDs if movie_id != movieID]  # remove the provided MovieID

    # Build a list of pairs MovieID (provided), MovieID to compare, association value
    associations = []
    for movie_id in movieIDs:  # for each MovieID compute the simple association and put everything in the list
        associations.append((movieID, movie_id, simple_association(dataset, movieID, movie_id)))

    # Sort pairs in descending order based on the association value and in case of ties based on the MovieID
    associations.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Retrieve the Top-N associations
    topN = [(tuple_[1], tuple_[2]) for tuple_ in associations[:n]]

    # Build the ranking
    ranking = {i + 1: topN[i] for i in range(len(topN))}

    return ranking


def topN_advanced(dataset, movieID, n):
    """
        Function that provides the ranking of the top N MovieIDs based on advanced association with the given MovieID

        Parameters:
        dataset (DataFrame): The dataset to manipulate.
        movieID (int): The query MovieID.
        n (int): The number of the best associations to find.

        Returns:
        dict: The ranking of the best n associations with movieID.
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    movieIDs = dataset["MovieID"].unique()  # get all MovieIDs
    movieIDs = [movie_id for movie_id in movieIDs if movie_id != movieID]  # remove the provided MovieID

    # Build a list of pairs MovieID (provided), MovieID to compare, association value
    associations = []
    for movie_id in movieIDs:  # for each MovieID compute the advanced association and put everything in the list
        associations.append((movieID, movie_id, advanced_association(dataset, movieID, movie_id)))

    # Sort pairs in descending order based on the association value and in case of ties based on the MovieID
    associations.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Retrieve the Top-N associations
    topN = [(tuple_[1], tuple_[2]) for tuple_ in associations[:n]]

    # Build the ranking
    ranking = {i + 1: topN[i] for i in range(len(topN))}

    return ranking


def topN_frequency(dataset, n, stars=None):
    """
        Function that provides a list of the top N movies based on rating frequency and optionally minimum number of
        stars.

        Parameters:
        dataset (DataFrame): The dataset to manipulate.
        n (int): The number of the most rated MovieIDs to find.
        stars (int): The minimum number of stars for the movie to be considered in the calculation.

        Returns:
        list: The most n rated movies, optionally with the minimum number of stars specified
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    filtered_dataset = dataset

    # Filter the dataset based on the condition Rating >= stars
    if stars is not None:
        filtered_dataset = dataset[dataset["Rating"] >= stars]

    # Count the frequency of each MovieID in descending order
    movie_frequencies = filtered_dataset["MovieID"].value_counts()

    # Extract the top n most rated MovieIDs
    topN = movie_frequencies.head(n)

    # Put them in a list
    topN_list = list(zip(topN.index, topN.values))

    return topN_list


def findNumUsers(dataset, movieID):
    """
        Function that calculates the number of users who wrote a review for a specified movieID.

        Parameters:
        dataset (DataFrame): The dataset where to perform the search.
        movieID (int): The MovieID for which to calculate the number of users.

        Returns:
        int: the number of users who wrote a review for the given movieID.
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    # Filter the dataset for the specified movieID
    movie_ratings = dataset[dataset["MovieID"] == movieID]

    # Count the number of unique users who rated the movie
    num_users = movie_ratings["UserID"].nunique()

    return num_users


def findTitle(dataset, movieID):
    """
        Function that finds the title of a movie

        Parameters:
        dataset (DataFrame): The dataset where to perform the search.
        movieID (int): The MovieID whose title we seek.

        Returns:
        str: The title of the movie
    """
    # Safety checks
    if len(dataset.columns) != len(movies_headers) or not all(dataset.columns == movies_headers):
        raise NotImplementedError("Dataset is not the movies")

    # Use loc to access rows by given labels, iloc to access the values at a given index
    return dataset.loc[dataset["MovieID"] == movieID, "Title"].iloc[0]


def calculate_frequency(dataset, movieID):
    """
        Function that finds the frequency of a movie

        Parameters:
        dataset (DataFrame): The dataset where to perform the search.
        movieID (int): The MovieID whose frequency we seek.

        Returns:
        int: The frequency of the movie
    """
    # Safety checks
    if len(dataset.columns) != len(ratings_headers) or not all(dataset.columns == ratings_headers):
        raise NotImplementedError("Dataset is not the ratings")

    # Filter dataset by considering rows with the given movieID and then return the number of rows
    return dataset[dataset['MovieID'] == movieID].shape[0]
