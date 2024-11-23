__author__ = "Dario Mameli"

from BaseRecommender import *


def printTopKNeighborsUser(tuples):
    """
        Print the top k user neighbors along with their similarity scores.

        Parameter:
        - tuples (list): A list of tuples containing user IDs and their similarity scores.

        Returns:
        - None: Prints the user IDs and similarity scores.
    """
    print(f"Top {len(tuples)} user neighbors:")
    print("UserID\tSimilarity")
    for tuple_ in tuples:
        print(f"{tuple_[0]}\t{tuple_[1]:.6f}")


class UUCFRecommender(BaseRecommender):
    """
        User-User Collaborative Filtering Recommender System.
    """

    def __init__(self, pathToMoviesDataset, pathToRatingsDataset):
        """
            Initialize the UUCFRecommender instance.

            Parameters:
                pathToMoviesDataset (str): The file path to the movies dataset.
                pathToRatingsDataset (str): The file path to the ratings dataset.
        """
        print("----------------"
              "USER USER COLLABORATIVE FILTERING"
              "----------------")
        super().__init__(pathToMoviesDataset, pathToRatingsDataset)

        # Parameter for tracking the significance weight of the last prediction
        self._gamma = None

    def _compute_average_rating(self, user_idx):
        """
            Compute the average rating for a given user.

            Parameters:
                user_idx (int): The index of the user in the ratings matrix.

            Returns:
                float: The average rating for the user. Returns 0 if the user has not rated any movies.
        """
        array = self._ratings_matrix[user_idx, self._ratings_matrix[user_idx, :] != 0]
        if array.size == 0:
            return 0
        mean = np.mean(array)
        return mean

    def pearson_correlation(self, user_a, user_u, gamma=10):
        """
            Compute the Pearson correlation coefficient between two users.

            Parameters:
                user_a (int): Index of user A in the ratings matrix.
                user_u (int): Index of user U in the ratings matrix.
                gamma (int, optional): A parameter for significance weighting. Defaults to 10.

            Returns:
                float: The Pearson correlation coefficient between user A and user U.
        """
        index_a = self._user_ids_lookup_table[user_a]
        index_u = self._user_ids_lookup_table[user_u]

        ra_avg = self._compute_average_rating(index_a)
        ru_avg = self._compute_average_rating(index_u)

        num = 0
        den_a = 0
        den_u = 0

        overlap = 0

        for movie_id in self._movie_ids:
            index_i = self._movie_ids_lookup_table[movie_id]
            r_ai = self._ratings_matrix[index_a, index_i]
            r_ui = self._ratings_matrix[index_u, index_i]

            if r_ai == 0 or r_ui == 0:
                continue

            overlap += 1

            num += (r_ai - ra_avg) * (r_ui - ru_avg)
            den_a += (r_ai - ra_avg) ** 2
            den_u += (r_ui - ru_avg) ** 2

        den_a = np.sqrt(den_a)
        den_u = np.sqrt(den_u)

        if overlap <= 1 or den_u == 0 or den_a == 0:
            return 0

        weight = num / (den_a * den_u)

        factor = 1
        if gamma is not None:
            Ra = np.nonzero(self._ratings_matrix[index_a, :])[0]
            Ru = np.nonzero(self._ratings_matrix[index_u, :])[0]
            intersection = np.intersect1d(Ra, Ru)
            factor = min(gamma, len(intersection)) / gamma

        return weight * factor

    def neighbors(self, user_a, movie_i, gamma=10):
        """
            Find the neighbors of a given user based on their similarity in ratings for a specific movie.

            Parameters:
                user_a (int): The index of the user for whom neighbors are being computed.
                movie_i (int): The index of the movie for which ratings are being considered.
                gamma (int, optional): A parameter for significance weighting. Defaults to 10.

            Returns:
                List[Tuple[int, float]]: A list of tuples containing user IDs and their corresponding similarity scores.
        """
        filtered_dataset = self._dataset_ratings[self._dataset_ratings['MovieID'] == movie_i]
        users = filtered_dataset['UserID'].to_list()

        tuples = []
        for user in users:
            if user != user_a:
                sim = self.pearson_correlation(user_a, user, gamma=gamma)
                if sim > 0:
                    tuples.append((int(user), float(sim)))

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        tuples.sort(key=lambda x: (-x[1], x[0]))

        return tuples

    def topK_neighbors(self, user_a, movie_i, k=20, gamma=10):
        """
            Find the top K neighbors of a given user based on their similarity in ratings for a specific movie.

            Parameters:
                user_a (int): The index of the user for whom top neighbors are being computed.
                movie_i (int): The index of the movie for which ratings are being considered.
                k (int, optional): The number of top neighbors to return. Defaults to 20.
                gamma (int, optional): A parameter for significance weighting. Defaults to 10.

            Returns:
                List[Tuple[int, float]]: A list of tuples containing user IDs and their corresponding similarity scores.
        """
        return self.neighbors(user_a, movie_i, gamma=gamma)[:k]

    def weighted_average_deviation_meanRating(self, user_a, movie_i):
        """
            Compute the weighted average of the deviation from the mean rating for a given user and movie.

            Parameters:
                user_a (int): The index of the user for whom the weighted average deviation is being computed.
                movie_i (int): The index of the movie for which the weighted average deviation is being computed.

            Returns:
                float: The weighted average of the deviation from the mean rating.
        """
        print("Computing weighted average of the deviation from mean rating..")
        prediction = self.predict_score(user_a, movie_i)
        index_a = self._user_ids_lookup_table[user_a]
        ra_avg = self._compute_average_rating(index_a)
        return prediction - ra_avg

    def predict_score(self, user_a, movie_i, gamma=10):
        """
            Predict the score that a given user would give to a given movie.

            Parameters:
                user_a (int): The index of the user for whom the score is being predicted.
                movie_i (int): The index of the movie for which the score is being predicted.
                gamma (int, optional): A parameter used in the computation of the weighted average deviation.
                Defaults to 10.

            Returns:
                float: The predicted score for the given movie by the user.
        """
        index_i = self._movie_ids_lookup_table[movie_i]
        index_a = self._user_ids_lookup_table[user_a]

        ra_avg = self._compute_average_rating(index_a)

        neighbors = self.topK_neighbors(user_a, movie_i, gamma=gamma)

        if not neighbors:
            return ra_avg

        num = 0
        den = 0

        for tuple_u in neighbors:
            user_u = tuple_u[0]
            sim = tuple_u[1]
            index_u = self._user_ids_lookup_table[user_u]
            ru_avg = self._compute_average_rating(index_u)
            r_ui = self._ratings_matrix[index_u, index_i]

            num += (r_ui - ru_avg) * sim
            den += sim

        return ra_avg + num / den

    def topN(self, userID, n=10, gamma=10):
        """
            Compute the top N movie recommendations for a given user.

            Parameters:
                userID (int): The ID of the user for whom recommendations are being computed.
                n (int, optional): The number of recommendations to return. Defaults to 10.
                gamma (int, optional): A parameter used in the computation of the recommendations. Defaults to 10.

            Returns:
                List[Tuple[int, str, float]]: A list of tuples containing the top N movie recommendations for the user.
                Each tuple contains the MovieID, Title, and predicted score.
        """
        # Checks to avoid unnecessary recomputation.
        if self._current_user == userID:
            if self._predictions is not None:
                if self._gamma == gamma:
                    print(f"Predictions for user {userID} with gamma {gamma} already computed.")
                    print(f"Printing top {n}..")
                    topN = self._predictions[:n]
                    printTopN(topN)
                    return topN

        print(f"Computing top {n} for user {userID} with gamma {gamma}..")
        filtered_dataset = self._dataset_ratings[self._dataset_ratings['UserID'] == userID]
        rated_movies = filtered_dataset['MovieID'].to_list()

        # Where to save the predictions
        tuples = []

        # Variables to track the progress of execution
        num_movies = len(self._movie_ids)
        start_time = time.time()
        num_iter_to_print = 100
        elapsed_times = []
        window = 5

        for i, movieID in enumerate(self._movie_ids):
            # Print information about execution progress and estimated completion.
            if i != 0 and i % num_iter_to_print == 0:
                remaining_movies = num_movies - i
                elapsed_time = time.time() - start_time
                elapsed_times.append(elapsed_time)
                print(f"Remaining movie predictions for user {userID}: {remaining_movies}")
                print(f"Computing time for {num_iter_to_print} movies: {elapsed_time:.2f} seconds")
                times = elapsed_times[-window:]
                average_time_per_movie = sum(times) / (len(times) * num_iter_to_print)
                estimated_remaining_time = average_time_per_movie * remaining_movies / 60
                if estimated_remaining_time >= 1:
                    print(f"Estimated remaining time for completion: "
                          f"{estimated_remaining_time:.0f} minutes")
                else:
                    print(f"Estimated remaining time for completion: "
                          f"{estimated_remaining_time * 60:.0f} seconds")
                start_time = time.time()

            # Compute the predictions
            if movieID not in rated_movies:
                # Retrieve info about the movie
                title = self._dataset_movies.loc[self._dataset_movies['MovieID'] == movieID, 'Title'].values[0]
                score = self.predict_score(userID, movieID, gamma=gamma)
                tuples.append((int(movieID), title, float(score)))

        print("Predictions computed.")

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        tuples.sort(key=lambda x: (-x[2], x[0]))

        # Save the state of the machine
        self._current_user = userID
        self._predictions = tuples
        self._gamma = gamma

        # Print and return the top N
        topN = tuples[:n]

        print("Done.")

        print(f"Printing top {n}..")
        printTopN(topN)

        return topN

    def get_predictions(self, userID, n):
        """
            Get all the movie predictions for a given user.
            If predictions are already computed for the specified user, returns the cached predictions.
            Otherwise, computes and returns new predictions using topN function.

            Parameters:
                userID (int): The ID of the user for whom predictions are being fetched.
                n (int): The number of predictions to compute in the top n.

            Returns:
                List[Tuple[int, str, float]]: A list of tuples containing the top N movie predictions for the user.
                Each tuple contains the MovieID, Title, and predicted score.
        """
        if self._predictions is not None:
            if userID == self._current_user:
                print("Predictions already computed.")
                return self._predictions
        self.topN(userID, n)
        return self._predictions
