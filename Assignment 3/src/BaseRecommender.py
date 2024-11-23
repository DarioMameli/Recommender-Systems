__author__ = "Dario Mameli"

import math
import pandas as pd
import numpy as np
from collections import Counter
from numpy.linalg import norm
import time


def printTopN(tuples):
    """
        Print the top n tuples with movie IDs titles and prediction scores.

        Parameter:
        - tuples (list): A list of tuples containing movie IDs titles and prediction scores.

        Returns:
        - None: Prints the top n tuples with movie IDs titles and prediction scores.
    """
    print(f"Top {len(tuples)} recommendations:")
    print("MovieID\tTitle\tPrediction")
    for tuple_ in tuples:
        print(f"{tuple_[0]}\t{tuple_[1]}\t{tuple_[2]:.6f}")


class BaseRecommender:
    """
        The base recommender containing the functions shared by all recommenders.
    """

    def __init__(self, pathToMoviesDataset, pathToRatingsDataset):
        """
            Initialize the BaseRecommender with datasets.

            Parameters:
            - pathToMoviesDataset (str): The path to the movies' dataset.
            - pathToRatingsDataset (str): The path to the ratings' dataset.
        """
        self.ratings_headers = ["UserID", "MovieID", "Rating", "Timestamp"]
        self.movies_headers = ["MovieID", "Title", "Genres"]

        # Dataframes containing the datasets
        self._dataset_movies = self.__read_dataset(pathToMoviesDataset)
        self._dataset_ratings = self.__read_dataset(pathToRatingsDataset)
        # Merge the two DataFrames on 'MovieID'
        self._dataset = pd.merge(self._dataset_ratings.drop(columns=['Timestamp']),
                                 self._dataset_movies, on='MovieID', how='left')

        # Get the lists of genres and userIDs in order, and movieIDs
        self._movie_ids = self.get_movie_ids()
        self._user_ids = self.get_user_ids()

        # Build the lookup tables to index the user and movie matrices
        self._movie_ids_lookup_table = dict(zip(self._movie_ids, range(len(self._movie_ids))))
        self._user_ids_lookup_table = dict(zip(self._user_ids, range(len(self._user_ids))))

        # Compute the matrix with all ratings for each user movie pair
        self.__compute_ratings_matrix()

        # Variables that store the last predictions for which user, to avoid expensive recomputations.
        self._predictions = None
        self._current_user = None

    # DATA LOADING ----------------------------------------------------------------------------------------------------

    def __read_dataset(self, path):
        """
            Function that reads the dataset from a specified path and stores it in a DataFrame

            Parameters:
            path (string): The path to the dataset.

            Returns:
            DataFrame: The dataset in a suitable format for manipulation.
        """
        print(f"Reading dataset at: {path}")
        dataset = pd.read_csv(path)
        substrings = path.split("/")
        for token in substrings:  # check which dataset we are loading and assign the corresponding column names
            if token == "ratings.csv":
                dataset.columns = self.ratings_headers
            elif token == "movies.csv":
                dataset.columns = self.movies_headers
                # Split the genres string into a list of genres
                dataset['Genres'] = dataset['Genres'].str.split('|')

        return dataset

    # GETTER FUNCTIONS --------------------------------------------------------------------------------------------

    def get_user_ids(self):
        """
            Get a list of unique user IDs.

            Returns:
            - list of str: A sorted list of unique user IDs.
        """
        # Get a list of unique UserIDs
        list_ = self._dataset_ratings['UserID'].unique().tolist()

        return list_

    def get_movie_ids(self):
        """
            Get a list of unique movie IDs.

            Returns:
            - list of str: A sorted list of unique movie IDs.
        """
        # Get a list of unique UserIDs
        list_ = self._dataset_movies['MovieID'].unique().tolist()

        return list_

    def get_title(self, movieID):
        """
            Get the title of a movie with ID movieID.

            Returns:
            - str: A string with the title of the movie.
        """
        filtered_dataset = self._dataset_movies[self._dataset_movies['MovieID'] == movieID]

        return filtered_dataset['Title'].values[0]

    def get_predictions(self, userID, n):
        """
            Get the last computed predictions given the user ID. n is for computing the top n if needed.

            Returns:
            - list of tuple: A list with all the tuple recommendations.
        """
        pass

    # MAIN FUNCTIONS -----------------------------------------------------------------------------------------------

    def retrieve_rating_history(self, userID, max_num=np.inf):
        """
            Retrieve the rating history of a specific user, including the movie ID, title, genre(s),
            and the original rating given by the user for each rated movie.

            Parameters:
            - userID (int): The ID of the user.

            Returns:
            None
        """
        # Merge the two DataFrames on 'MovieID'
        filtered_dataset = self._dataset[self._dataset['UserID'] == userID]
        print(f"Rating history of user {userID}:")
        print("MovieID\tTitle\tGenres\tRating (orig.)")
        i = 0
        for _, row in filtered_dataset.iterrows():
            if i >= max_num:
                break
            movieID = row['MovieID']
            title = row['Title']
            genres = row['Genres']
            rating = row['Rating']
            print(f"{movieID}\t{title}\t{genres}\t{rating}")
            i += 1

    def __compute_ratings_matrix(self):
        """
            Compute the ratings matrix from the ratings dataset.

            Returns:
            - None: The ratings matrix is stored internally in the object.
        """
        print("Computing ratings matrix..")
        ratings_matrix = np.zeros((len(self._user_ids), len(self._movie_ids)), dtype=float)  # Initialization
        n = len(self._user_ids)
        for i, u in enumerate(self._user_ids):  # For each user
            if i % 50 == 0:
                print(f"\tRemaining users {n - i}..")
            indexUser = self._user_ids_lookup_table[u]  # Retrieve the index of user
            filtered_dataset = self._dataset_ratings[self._dataset_ratings['UserID'] == u]  # Filter the dataset
            for _, row in filtered_dataset.iterrows():  # For each row
                # Retrieve the information
                movieID = int(row['MovieID'])
                rating = float(row['Rating'])
                indexMovie = self._movie_ids_lookup_table[movieID]  # Retrieve the index of movie
                ratings_matrix[indexUser, indexMovie] += rating  # Update the ratings matrix
        self._ratings_matrix = ratings_matrix  # Save the ratings matrix
        print("Done.")
