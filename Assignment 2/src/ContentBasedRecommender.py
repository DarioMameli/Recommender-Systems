__author__ = "Dario Mameli"

import math
import pandas as pd
import numpy as np
from collections import Counter
from numpy.linalg import norm


def cosine_similarity(A, B):
    """
        Compute the cosine similarity between two vectors.

        Parameters:
        - A (numpy.ndarray): The first vector.
        - B (numpy.ndarray): The second vector.

        Returns:
        float: The cosine similarity between vectors A and B.
    """
    # Check if norms are zero
    if norm(A) == 0:
        print("norm(A)", norm(A))
    if norm(B) == 0:
        print("norm(B)", norm(B))

    # Compute the dot product
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return cosine


def cosine_similarity_arr(A, B_array):
    """
        Compute the cosine similarity between a vector A and an array of vectors B.

        Parameters:
        - A (numpy.ndarray): The first vector.
        - B_array (list of numpy.ndarray): The array of vectors.

        Returns:
        float: The maximum cosine similarity between vector A and vectors in B_array.
               Returns 0 if B_array is empty.
    """
    # Check if array is empty
    if not B_array:
        return 0

    # Return maximum similarity
    similarities = [cosine_similarity(A, B) for B in B_array]
    return max(similarities)


class ContentBasedRecommender:
    """
        A content-based recommender system that suggests movies to users based on the genre of movies and the rating
        history of the users.

        Attributes:
        - ratings_headers (list of str): The headers for ratings data.
        - movies_headers (list of str): The headers for movies data.
    """

    ratings_headers = ["UserID", "MovieID", "Rating", "Timestamp"]
    movies_headers = ["MovieID", "Title", "Genres"]

    def __init__(self, pathToMoviesDataset, pathToRatingsDataset):
        """
            Initialize the ContentBasedRecommender with datasets.

            Parameters:
            - pathToMoviesDataset (str): The path to the movies' dataset.
            - pathToRatingsDataset (str): The path to the ratings' dataset.
        """
        print("--------------------------------------- "
              "CONTENT BASED RECOMMENDER "
              "---------------------------------------")

        # Dataframes containing the datasets
        self.__dataset_movies = self.__read_dataset(pathToMoviesDataset)
        self.__dataset_ratings = self.__read_dataset(pathToRatingsDataset)
        # Merge the two DataFrames on 'MovieID'
        self.__dataset = pd.merge(self.__dataset_ratings.drop(columns=['Timestamp']),
                                  self.__dataset_movies, on='MovieID', how='left')

        # Get the lists of genres and userIDs in order, and movieIDs
        self.__genres = self.get_genres()
        self.__movie_ids = self.get_movie_ids()
        self.__user_ids = self.get_user_ids()

        # Build the lookup tables to index the user and movie matrices
        self.__genres_lookup_table = dict(zip(self.__genres, range(len(self.__genres))))
        self.__movie_ids_lookup_table = dict(zip(self.__movie_ids, range(len(self.__movie_ids))))
        self.__user_ids_lookup_table = dict(zip(self.__user_ids, range(len(self.__user_ids))))

        # Initialize the user profiles, movies encoded, score matrices, IDF dictionary and ranking dataframe
        self.__user_profiles = None
        self.__movies_encoded = None
        self.__scores = None
        self.__IDF_dict = None
        self.__ranking = None

        # Boolean variables to keep track of the state of the matrices
        self.__normalization = False
        self.__rescale = False

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
        dataset = pd.read_csv(path, sep=',', header=None, engine='python', encoding='utf-8')
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

    def get_genres(self):
        """
            Get a list of unique movie genres.

            Returns:
            - list of str: A sorted list of unique movie genres.
        """
        # Explode the 'Genres' column to create one row for each genre
        exploded_genres = self.__dataset_movies['Genres'].explode()

        # Get unique genres
        unique_genres = exploded_genres.unique()

        # Convert unique genres to a list
        list_ = unique_genres.tolist()

        # Remove useless tag
        list_.remove(list_[0])

        # Sort the list
        list_.sort()

        return list_

    def get_user_ids(self):
        """
            Get a list of unique user IDs.

            Returns:
            - list of str: A sorted list of unique user IDs.
        """
        # Get a list of unique UserIDs
        list_ = self.__dataset_ratings['UserID'].unique().tolist()

        # Remove useless tag
        list_.remove(list_[0])

        return list_

    def get_movie_ids(self):
        """
            Get a list of unique movie IDs.

            Returns:
            - list of str: A sorted list of unique movie IDs.
        """
        # Get a list of unique UserIDs
        list_ = self.__dataset_movies['MovieID'].unique().tolist()

        # Remove useless tag
        list_.remove(list_[0])

        return list_

    def get_user_profile(self, userID, movies_normalization=False, users_rescaling=False):
        """
            Get the profile of a user.

            Parameters:
            - userID (str): The ID of the user whose profile is to be retrieved.
            - movies_normalization (bool): Whether to normalize the movie features in the user profile.
            - users_rescaling (bool): Whether to rescale the user profile features using IDF.

            Returns:
            - None: This method only prints the profile of the user.
        """
        # Compute the users profiles
        self.__compute_user_profiles(movies_normalization, users_rescaling)

        # Retrieve the user profile from the matrix
        userID = str(userID)
        print(f"Profile of user {userID}:")
        index = self.__user_ids_lookup_table[userID]
        user_profile = self.__user_profiles[index]

        # Print the user profile
        print("Genre\tScore")
        for i in range(len(user_profile)):
            print(f"{self.__genres[i]}\t{user_profile[i]:.4f}")

    # MAIN FUNCTIONS -------------------------------------------------------------------------------------------------

    def __compute_user_profiles(self, normalization=False, rescale=False):
        """
            Compute user profiles based on their rated movies.

            Parameters:
            - normalization (bool): Whether to normalize the movie features before computing user profiles.
            - rescale (bool): Whether to rescale the user profiles using IDF.

            Returns:
            - None: This method computes user profiles and updates the internal state of the object but does not return
            any value.
        """
        # Conditions for avoiding redundant computation
        if not self.__compute_movies_encoded(normalization):
            if rescale != self.__rescale:
                if rescale:
                    if self.__user_profiles is not None:
                        self.__rescale_user_profiles_with_IDF()
                        return
            else:
                print(f"User profiles with rescale={rescale} already computed..")
                return

        # Compute user profiles
        print(f"Computing user profiles with rescale={rescale}..")
        user_profiles = np.zeros((len(self.__user_ids), len(self.__genres)), dtype=float)  # Initialization
        for u in self.__user_ids:  # For each user
            indexUser = self.__user_ids_lookup_table[u]  # Retrieve the index of user
            filtered_dataset = self.__dataset_ratings[self.__dataset_ratings['UserID'] == u]  # Filter the dataset
            for _, row in filtered_dataset.iterrows():  # For each row
                # Retrieve the information
                movieID = row['MovieID']
                rating = float(row['Rating']) - 3
                indexMovie = self.__movie_ids_lookup_table[movieID]  # Retrieve the index of movie
                user_profiles[indexUser] += self.__movies_encoded[indexMovie] * rating  # Compute the user profile
        self.__user_profiles = user_profiles  # Save the user profiles
        # Rescale if necessary
        if rescale:
            self.__rescale_user_profiles_with_IDF()
        else:
            self.__rescale = False

    def __compute_movies_encoded(self, normalization=False):
        """
            Compute the encoded representation of movies based on their genres.

            Parameters:
            - normalization (bool): Whether to normalize the movie features.

            Returns:
            - bool: True if the computation was performed, False if the matrix was already computed with the same
            normalization.
        """
        # Avoid redundant computation
        if self.__movies_encoded is not None:
            if self.__normalization == normalization:
                print(f"Movies matrix with normalization={normalization} already computed..")
                return False

        # Computing movie matrix
        print(f"Computing movie matrix with normalization={normalization}..")
        movies_encoded = np.zeros((len(self.__movie_ids), len(self.__genres)), dtype=float)  # Initialization
        for m in self.__movie_ids:  # For each movie
            filtered_dataset = self.__dataset_movies[self.__dataset_movies['MovieID'] == m]  # Filter dataset
            for _, row in filtered_dataset.iterrows():  # For each row
                genres = row['Genres']  # Retrieve the genres
                if normalization:  # If movie normalization is needed
                    den = math.sqrt(len(genres))
                else:
                    den = 1
                for genre in genres:  # For each genre
                    # Retrieve indices
                    indexMovie = self.__movie_ids_lookup_table[m]
                    indexGenre = self.__genres_lookup_table[genre]
                    movies_encoded[indexMovie, indexGenre] += 1 / den  # Compute the normalized movie
                break

        # Checks on the matrix correct encodings
        if normalization:
            if not (movies_encoded <= 1).all():
                raise ValueError("Matrix should have values lower or equal to 1")
        else:
            if not np.logical_or(movies_encoded == 1, movies_encoded == 0).all():
                raise ValueError("Matrix should only have zeros or ones")

        self.__movies_encoded = movies_encoded
        self.__normalization = normalization
        return True

    def IDF(self):
        """
            Compute the Inverse Document Frequency (IDF) for each genre.

            Returns:
            - dict: A dictionary containing the IDF value for each genre.
        """
        print("Computing IDF..")
        # Explode the 'Genres' column to create one row for each genre
        exploded_genres = self.__dataset_movies['Genres'].explode()

        # Filter out initial tag
        exploded_genres = exploded_genres[exploded_genres != 'genres']

        # Sort values
        exploded_genres = exploded_genres.sort_values()

        # Count occurrences of each genre
        genre_counts = Counter(exploded_genres)

        # Build the IDF dictionary
        IDF_dict = dict()
        for genre, count in genre_counts.items():
            IDF_dict.update({genre: 1 / count})

        return IDF_dict

    def __rescale_user_profiles_with_IDF(self):
        """
            Rescale the user profiles using the Inverse Document Frequency (IDF) values.

            Returns:
            - None
        """
        # Build the IDF dictionary if not present
        if self.__IDF_dict is None:
            self.__IDF_dict = self.IDF()

        print("Rescaling user profiles..")
        for i in range(self.__user_profiles.shape[1]):  # For each genre
            column = self.__user_profiles[:, i]  # Retrieve the score for the genre in the profiles matrix
            column = column * self.__IDF_dict[self.__genres[i]]  # Rescale using IDF
            self.__user_profiles[:, i] = column  # Update the column
        self.__rescale = True

    def __compute_scores(self, movies_normalization=False, users_rescaling=False):
        """
            Compute the scores by taking the dot product of the user profiles and the encoded movie matrix.

            Parameters:
            - movies_normalization (bool): Whether to normalize movie features before computing scores.
            - users_rescaling (bool): Whether to rescale user profiles before computing scores.

            Returns:
            - None
        """
        print("Computing scores..")
        self.__compute_user_profiles(movies_normalization, users_rescaling)
        print("Dot product..")
        # NOTE: Don't really need a function avoid redundant computation as the dot product is quite efficient anyway
        self.__scores = self.__user_profiles.dot(self.__movies_encoded.transpose())

    def __rank(self, userID, n, movies_normalization, users_rescaling):
        """
            Ranks the best n movies for a given user based on scores.

            Parameters:
            - userID (str): The ID of the user for whom to rank movies.
            - n (int): The number of movies to rank.
            - movies_normalization (bool): Whether to normalize movie features before computing scores.
            - users_rescaling (bool): Whether to rescale user profiles before computing scores.

            Returns:
            - None
        """
        print(f"Ranking {n} movies for user {userID}..")

        # First compute the scores for the user
        self.__compute_scores(movies_normalization, users_rescaling)
        movie_scores = self.__scores[self.__user_ids_lookup_table[userID], :]

        # Retrieve the corresponding movieIDs
        indices = np.argsort(movie_scores)[::-1]
        movie_scores = np.sort(movie_scores)[::-1]
        movieIDs = [self.__movie_ids[i] for i in indices]

        # Filter the dataset of ratings for the user
        filtered_dataset = self.__dataset_ratings[self.__dataset_ratings['UserID'] == userID]
        movies_of_user = filtered_dataset['MovieID'].values

        # Remove movies which have already been rated by the user
        movieIDs_tmp = []
        movie_scores_tmp = []
        for i in range(len(movieIDs)):
            if movieIDs[i] not in movies_of_user:
                movieIDs_tmp.append(movieIDs[i])
                movie_scores_tmp.append(movie_scores[i])
        movie_scores = movie_scores_tmp
        movieIDs = movieIDs_tmp

        # Create a list of tuples containing movieID, movie_score, title, and genres
        data = []
        for movieID, movie_score in zip(movieIDs, movie_scores):
            filtered_movies = self.__dataset_movies[self.__dataset_movies['MovieID'] == movieID]
            title = filtered_movies['Title'].values[0]
            genres = filtered_movies['Genres'].values[0]
            data.append((int(movieID), title, genres, float(movie_score)))

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        data.sort(key=lambda x: (-x[3], x[0]))

        # Create a ranking DataFrame from the collected data
        self.__ranking = pd.DataFrame(data, columns=['MovieID', 'Title', 'Genres', 'Score']).head(n)
        print("Ranking completed.")

    def __printTopN(self, n):
        """
            Print the top N ranked movies.

            This method prints the top N ranked movies along with their IDs, titles, genres, and scores.

            Parameters:
            - n (int): The number of top-ranked movies to print.

            Returns:
            - None
        """
        rank = 0
        print("Rank\tID\tTitle\tGenres\tScore")
        for _, row in self.__ranking.head(n).iterrows():  # Taking the head n is not needed as the ranking is already
            # of the top N, however it is just as a safety measure
            rank += 1
            movieID = row['MovieID']
            title = row['Title']
            genres = row['Genres']
            score = row['Score']
            print(f"{rank}\t{movieID}\t{title}\t{genres}\t{score:.4f}")

    def topN(self, userID, n, movies_normalization=False, users_rescaling=False):
        """
            Generate top N recommendations for a user and print them.

            Parameters:
            - userID (str): The ID of the user for whom to generate recommendations.
            - n (int): The number of recommendations to generate.
            - movies_normalization (bool): Whether to normalize movie features before generating recommendations.
            - users_rescaling (bool): Whether to rescale user profiles before generating recommendations.

            Returns:
            - None
        """
        userID = str(userID)
        print(f"Computing Top {n} for user {userID}..")
        self.__rank(userID, n, movies_normalization, users_rescaling)
        self.__printTopN(n)

    def topN_rerank(self, userID, n, n_in=200, lambda_=0.5, movies_normalization=False, users_rescaling=False):
        """
            Generate top N reranked recommendations for a user using Maximal Marginal Relevance (MMR) reranking.
            It first ranks the top n_in movies for the user using the __rank method, then reranks them using MMR
            algorithm.
            The MMR algorithm iteratively selects the most relevant and diverse items based on a given lambda parameter.

            Parameters:
            - userID (str): The ID of the user for whom to generate reranked recommendations.
            - n (int): The number of reranked recommendations to generate.
            - n_in (int): The number of initial recommendations to consider for reranking.
            - lambda_ (float): The trade-off parameter controlling the balance between relevance and diversity in MMR.
            - movies_normalization (bool): Whether to normalize movie features before reranking.
            - users_rescaling (bool): Whether to rescale user profiles before reranking.

            Returns:
            - None
        """

        # Compute initial ranking
        userID = str(userID)
        print(f"Computing Top {n} with MMR reranking for user {userID}..")
        self.__rank(userID, n_in, movies_normalization, users_rescaling)

        # Compute the input related items
        index_of_userID = self.__user_ids_lookup_table[userID]
        query = self.__user_profiles[index_of_userID]
        relatedItems = []
        for i, row in self.__ranking.iterrows():
            movieID = str(row['MovieID'])
            index_of_movieID = self.__movie_ids_lookup_table[movieID]
            item = self.__movies_encoded[index_of_movieID]
            # Use tuples (index, item) to retrieve the indices in the ranking more easily
            relatedItems.append((i, item))

        # MMR Algorithm -----------------------------------------------
        reranking_data = []  # Stores the top n rows of the ranking in order
        diversifiedItems = []
        rank = 0
        while len(diversifiedItems) < n:
            rank += 1
            print(f"Computing MMR Rank: {rank}..")
            bestItem = None
            bestScore = -np.inf
            index_in_ranking = None
            for i, tuple_ in enumerate(relatedItems):
                j = tuple_[0]
                item = tuple_[1]
                relevanceToQuery = cosine_similarity(item, query)
                similarityWithSelectedItems = cosine_similarity_arr(item, diversifiedItems)
                combinedScore = lambda_ * relevanceToQuery - (1 - lambda_) * similarityWithSelectedItems
                if combinedScore > bestScore:
                    bestItem = item  # Save the best item
                    bestScore = combinedScore  # Save the MMR score of the best item
                    index_in_ranking = j  # Save the index in the ranking of the best item
            diversifiedItems.append(bestItem)
            relatedItems.remove((index_in_ranking, bestItem))
            row = self.__ranking.loc[index_in_ranking].copy()  # Retrieve the row in the initial ranking
            row['Score'] = bestScore  # Assign the recalculated score
            reranking_data.append(row)

        # Build the new ranking
        self.__ranking = pd.DataFrame(reranking_data, columns=['MovieID', 'Title', 'Genres', 'Score'])
        self.__printTopN(n)

    def compute_genre_frequency(self, genre):
        """
            Compute the frequency of a specific genre in the movie dataset.

            Parameters:
            - genre (str): The genre for which to compute the frequency.

            Returns:
            - int: The frequency of movies belonging to the specified genre.
        """

        # Explode the 'Genres' column to create a new row for each genre
        exploded_genres = self.__dataset_movies.explode('Genres')

        # Filter the DataFrame to include only rows with the specified genre
        genre_occurrences = exploded_genres[exploded_genres['Genres'] == genre]

        # Count the frequency of the specified genre
        return len(genre_occurrences)

    def compute_min_max_average_genres(self):
        """
            Compute the minimum, maximum, and average number of genres per movie.

            Returns:
            - tuple: A tuple containing the minimum number of genres per movie,
                     the maximum number of genres per movie, and the average number of genres per movie.
        """
        lists_of_genres = self.__dataset_movies['Genres']
        count = 0
        min_ = np.inf
        max_ = -1
        for list_ in lists_of_genres:
            if list_ == ['genres']:
                continue
            num_genres = len(list_)
            count += num_genres
            if num_genres < min_:
                min_ = num_genres
            if num_genres > max_:
                max_ = num_genres
        return min_, max_, count / len(self.__movie_ids)

    def retrieve_rating_info(self, userID, movieID):
        """
            Retrieve rating information for a specific user and movie, including the ID of the movie, the title of the
            movie, its genre(s), the rating given by the user, and the date and time of the rating.

            Parameters:
            - userID (int or str): The ID of the user.
            - movieID (int or str): The ID of the movie.

            Returns:
            None
        """
        # Filter the ratings' dataset in search of occurrences of the user and movie
        userID = str(userID)
        movieID = str(movieID)
        rating_info = self.__dataset_ratings[(self.__dataset_ratings['UserID'] == userID) &
                                             (self.__dataset_ratings['MovieID'] == movieID)]
        # Get rating, date, and time
        rating = rating_info['Rating'].values[0]
        timestamp = int(rating_info['Timestamp'].values[0])
        date_time = pd.to_datetime(timestamp, unit='s')  # Convert timestamp to date and time

        # Filter the movies' dataset in search of occurrences of the movie
        metadata_movie = self.__dataset_movies[self.__dataset_movies['MovieID'] == movieID]
        # Get title and genre(s)
        title = metadata_movie['Title'].values[0]
        genres = metadata_movie['Genres'].values[0]

        print(f"UserID: {userID}\nMovieID: {movieID}\nTitle: {title}\nGenres: {genres}\nRating: {rating}\n"
              f"Date&Time: {date_time}")

    def retrieve_rating_history(self, userID):
        """
            Retrieve the rating history of a specific user, including the movie ID, title, genre(s),
            and the original rating given by the user for each rated movie.

            Parameters:
            - userID (int or str): The ID of the user.

            Returns:
            None
        """
        userID = str(userID)
        filtered_dataset = self.__dataset[self.__dataset['UserID'] == userID]
        print(f"Rating history of user {userID}:")
        print("MovieID\tTitle\tGenres\tRating (orig.)")
        for _, row in filtered_dataset.iterrows():
            movieID = row['MovieID']
            title = row['Title']
            genres = row['Genres']
            rating = row['Rating']
            print(f"{movieID}\t{title}\t{genres}\t{rating}")

    def compute_genres_frequencies(self, userID):
        """
            Compute the frequencies of genres in the rating history of a specific user, i.e. it counts how many times
            each genre appears in the movies rated by the user.

            Parameters:
            - userID (int or str): The ID of the user.

            Returns:
            None
        """
        userID = str(userID)
        filtered_dataset = self.__dataset[self.__dataset['UserID'] == userID]

        # Explode the 'Genres' column to create a new row for each genre
        all_genres = filtered_dataset['Genres'].explode()

        # Count the frequency of each genre
        genre_counts = all_genres.value_counts()

        # Print the genre frequencies
        print(f"Genre frequencies of user {userID}:")
        print(genre_counts)

    def study_ratings(self, userID):
        """
            Study the ratings given by a specific user.

            This method calculates various statistics about the ratings given by a specific user,
            including the average and median rating, as well as the share of positive, neutral, and negative ratings.

            Parameters:
            - userID (int or str): The ID of the user.

            Returns:
            None
        """
        userID = str(userID)
        filtered_dataset = self.__dataset[self.__dataset['UserID'] == userID]

        # Ratings rescaling
        ratings = pd.to_numeric(filtered_dataset['Rating'], errors='coerce')
        ratings = ratings - 3

        # Calculate average and median rating
        average_rating = ratings.mean()
        median_rating = ratings.median()

        # Calculate share of positive/neutral/negative ratings
        total_ratings = len(ratings)
        positive_ratings = len(ratings[ratings > 0])
        neutral_ratings = len(ratings[ratings == 0])
        negative_ratings = len(ratings[ratings < 0])

        # Print results
        print(f"Ratings study for user {userID}:")
        print(f"Average rating\t{average_rating:.2f}")
        print(f"Median rating\t{median_rating}")
        print(f"Share of positive ratings\t{positive_ratings / total_ratings:.2%}")
        print(f"Share of neutral ratings\t{neutral_ratings / total_ratings:.2%}")
        print(f"Share of negative ratings\t{negative_ratings / total_ratings:.2%}")

    def study_ratings_per_genre(self, userID):
        """
            Study the ratings given by a specific user, by simply calculating the average rating given by a
            user for each genre they have rated.

            Parameters:
            - userID (int or str): The ID of the user.

            Returns:
            None
        """
        userID = str(userID)
        filtered_dataset = self.__dataset[self.__dataset['UserID'] == userID]

        # Flatten the lists in the 'Genres' column (eliminating invalid entries)
        genres = filtered_dataset['Genres'].explode().dropna()

        # Calculate average rating for each genre
        genre_avg_ratings = {}
        for genre in genres.unique():
            # Filter dataset to only include rows where genre is present
            genre_ratings = filtered_dataset[filtered_dataset['Genres'].apply(lambda x: genre in x)]
            # Rescale the ratings
            ratings = pd.to_numeric(genre_ratings['Rating'], errors='coerce') - 3
            # Calculate the mean of the ratings in the genre_ratings dataframe
            avg_rating = ratings.mean()
            # Save the result
            genre_avg_ratings[genre] = avg_rating

        # Print results
        print(f"Ratings study per genre for user {userID}:")
        print("Genre\tAve. Rating")
        for genre, avg_rating in genre_avg_ratings.items():
            print(f"{genre}\t{avg_rating:.2f}")
