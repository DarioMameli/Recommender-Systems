__author__ = "Dario Mameli"

from BaseRecommender import *


def printTopKNeighborsMovie(tuples):
    print(f"Top {len(tuples)} movie neighbors:")
    print("MovieID\tSimilarity")
    for tuple_ in tuples:
        print(f"{tuple_[0]}\t{tuple_[1]:.6f}")


class IICFRecommender(BaseRecommender):
    """
        Item-Item Collaborative Filtering Recommender System.
    """

    def __init__(self, pathToMoviesDataset, pathToRatingsDataset, model_based=True):
        """
            Initialize the Item-Item Collaborative Filtering Recommender System.

            Parameters:
                pathToMoviesDataset (str): The file path to the movies dataset.
                pathToRatingsDataset (str): The file path to the ratings dataset.
                model_based (bool, optional): A flag indicating whether to use a model-based approach.
                    If True, the similarity model will be computed during initialization.
                    Defaults to True.
        """
        print("----------------"
              "ITEM ITEM COLLABORATIVE FILTERING"
              "----------------")
        super().__init__(pathToMoviesDataset, pathToRatingsDataset)

        # Build a model-based IICF recommender if requested
        self.model_based = model_based
        if model_based:
            self._compute_similarity_model()

    def get_similarity_model(self):
        """
            Get the similarity model.

            Returns:
                Union[pd.Dataframe, None]: The similarity model if available, otherwise None.
            """
        return self.similarity_model

    def _compute_average_rating(self, movie_idx):
        """
            Compute the average rating for a given movie.

            Parameter:
                movie_idx (int): The index of the movie.

            Returns:
                float: The average rating for the movie.
        """
        array = self._ratings_matrix[self._ratings_matrix[:, movie_idx] != 0, movie_idx]
        if array.size == 0:
            return 0
        return np.mean(array)

    def _compute_similarity_model(self):
        """
            Compute the model for item-item similarities.
        """
        print("Computing the model for the similarities..")
        dataset = self._dataset_ratings

        # Calculate mean ratings for each movie
        mean_ratings = dataset.groupby('MovieID')['Rating'].transform("mean")

        # Normalize ratings by subtracting mean ratings
        dataset['Norm_Rating'] = dataset['Rating'] - mean_ratings

        # Calculate squared normalized ratings
        dataset['Sqr_Norm_Rating'] = dataset['Norm_Rating'] ** 2

        # Gather data into pivot tables for efficient matrix multiplications
        numerator_table = pd.pivot_table(dataset, index='UserID', columns='MovieID', values='Norm_Rating', fill_value=0)
        denominator_table = pd.pivot_table(dataset, index='UserID', columns='MovieID', values='Sqr_Norm_Rating',
                                           fill_value=0)

        # Compute the numerator
        num = np.dot(numerator_table.T, numerator_table)

        # Sum the denominator rows over the user axis
        den_sum = denominator_table.sum(axis=0)

        # Calculate the outer product of den_sum with itself to get the resulting matrix
        den = np.outer(den_sum, den_sum)

        # Calculate the square root of each entry
        den = np.sqrt(den)

        # Compute similarity matrix
        similarity_matrix = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        # Set the indices of the similarity matrix to match the movie IDs
        similarity_matrix = pd.DataFrame(similarity_matrix, index=numerator_table.columns,
                                         columns=numerator_table.columns)

        print("Done.")

        self.similarity_model = similarity_matrix

    def cosine_similarity(self, movie_i, movie_j):
        """
            Compute the cosine similarity between two movies.

            Parameters:
            - movie_i (int): The ID of the first movie.
            - movie_j (int): The ID of the second movie.

            Returns:
            - float: The cosine similarity between the two movies.
        """
        index_i = self._movie_ids_lookup_table[movie_i]
        index_j = self._movie_ids_lookup_table[movie_j]

        ri_avg = self._compute_average_rating(index_i)
        rj_avg = self._compute_average_rating(index_j)

        num = 0
        den_i = 0
        den_j = 0

        for user_id in self._user_ids:
            index_u = self._user_ids_lookup_table[user_id]
            r_ui = self._ratings_matrix[index_u, index_i]
            r_uj = self._ratings_matrix[index_u, index_j]

            if r_ui != 0 and r_uj != 0:
                num += (r_ui - ri_avg) * (r_uj - rj_avg)
            if r_ui != 0:
                den_i += (r_ui - ri_avg) ** 2
            if r_uj != 0:
                den_j += (r_uj - rj_avg) ** 2

        den_i = np.sqrt(den_i)
        den_j = np.sqrt(den_j)

        if den_i == 0 or den_j == 0:
            return 0

        return num / (den_i * den_j)

    def neighbors(self, user_u, movie_i):
        """
            Find neighbors of a given movie for a given user.

            Parameters:
            - user_u (int): The ID of the user.
            - movie_i (int): The ID of the movie.

            Returns:
            - list of tuples: List of tuples containing the ID of the neighboring movie and its similarity score.
        """
        filtered_dataset = self._dataset_ratings[self._dataset_ratings['UserID'] == user_u]
        movies = filtered_dataset['MovieID'].to_list()

        tuples = []
        for movie in movies:
            if movie != movie_i:
                if self.model_based:
                    sim = self.similarity_model.loc[movie_i, movie]
                else:
                    sim = self.cosine_similarity(movie_i, movie)
                if sim > 0:
                    tuples.append((int(movie), float(sim)))

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        tuples.sort(key=lambda x: (-x[1], x[0]))

        return tuples

    def topK_neighbors(self, user_u, movie_i, k=20):
        """
            Find the top k neighbors of a given movie for a given user.

            Parameters:
            - user_u (int): The ID of the user.
            - movie_i (int): The ID of the movie.
            - k (int): The number of neighbors to retrieve (default is 20).

            Returns:
            - list of tuples: List of tuples containing the ID of the neighboring movie and its similarity score.
        """
        return self.neighbors(user_u, movie_i)[:k]

    def predict_score(self, user_u, movie_i):
        """
            Predict the score that a given user would give to a given movie.

            Parameters:
                user_u (int): The index of the user for whom the score is being predicted.
                movie_i (int): The index of the movie for which the score is being predicted.

            Returns:
                float: The predicted score for the given movie by the user.
        """
        index_u = self._user_ids_lookup_table[user_u]

        neighbors = self.topK_neighbors(user_u, movie_i)

        if not neighbors:
            return 0

        num = 0
        den = 0

        for tuple_j in neighbors:
            movie_j = tuple_j[0]
            sim = tuple_j[1]
            index_j = self._movie_ids_lookup_table[movie_j]
            r_uj = self._ratings_matrix[index_u, index_j]

            num += r_uj * sim
            den += abs(sim)

        return num / den

    def __score_basket(self, movie_i, basket, all_sim):
        """
            Calculate the score of a basket of movies with respect to a given movie.

            Parameters:
            - movie_i (int): The ID of the target movie.
            - basket (list of int): List of movie IDs in the basket.
            - all_sim (bool): Flag indicating whether to consider all similarities or only positive similarities.

            Returns:
            - float: The score of the basket.
        """
        if not basket:
            return 0
        sim = 0
        for movie_j in basket:
            tmp = self.cosine_similarity(movie_i, movie_j)
            if not all_sim:
                if tmp <= 0:
                    continue
            sim += tmp

        return sim

    def topN_basket(self, basket, all_sim=False, n=10):
        """
            Compute top N recommendations for a basket of movies.

            Parameters:
            - basket (list of int): List of movie IDs in the basket.
            - all_sim (bool): Flag indicating whether to consider all similarities or only positive similarities.
            - n (int): Number of recommendations to return.

            Returns:
            - list: List of top N recommendations.
        """
        print(f"Computing top {n} for basket {basket}..")

        # Where to save the predictions
        tuples = []

        # Variables to track the progress of execution
        num_movies = len(self._movie_ids)
        start_time = time.time()  # Initialize start_time outside the loop
        num_iter_to_print = 1000
        elapsed_times = []
        window = 5

        for i, movieID in enumerate(self._movie_ids):
            # Print information about execution progress and estimated completion.
            if i != 0 and i % num_iter_to_print == 0:
                remaining_movies = num_movies - i
                elapsed_time = time.time() - start_time
                elapsed_times.append(elapsed_time)
                print(f"Remaining movie predictions for basket {basket}: {remaining_movies}")
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
            if movieID not in basket:
                # Retrieve info about the movie
                title = self._dataset_movies.loc[self._dataset_movies['MovieID'] == movieID, 'Title'].values[0]
                score = self.__score_basket(movieID, basket, all_sim)
                tuples.append((int(movieID), title, float(score)))

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        tuples.sort(key=lambda x: (-x[2], x[0]))

        # Print and return the top N
        topN = tuples[:n]

        print("Done.")

        print(f"Printing top {n}..")
        printTopN(topN)

        return topN

    def topN(self, userID, n=10):
        """
            Compute the top N movie recommendations for a given user.

            Parameters:
                userID (int): The ID of the user for whom recommendations are being computed.
                n (int, optional): The number of recommendations to return. Defaults to 10.

            Returns:
                List[Tuple[int, str, float]]: A list of tuples containing the top N movie recommendations for the user.
                Each tuple contains the MovieID, Title, and predicted score.
        """
        # Checks to avoid unnecessary recomputation.
        if self._current_user == userID:
            if self._predictions is not None:
                print(f"Predictions for user {userID} already computed.")
                print(f"Printing top {n}..")
                topN = self._predictions[:n]
                printTopN(topN)
                return topN

        print(f"Computing top {n} for user {userID}..")
        filtered_dataset = self._dataset_ratings[self._dataset_ratings['UserID'] == userID]
        rated_movies = filtered_dataset['MovieID'].to_list()

        # Where to save the predictions
        tuples = []

        # Variables to track the progress of execution
        num_movies = len(self._movie_ids)
        start_time = time.time()  # Initialize start_time outside the loop
        total_elapsed_time = time.time() - start_time

        # Iterate through different sequences based on the algorithmic approach
        if not self.model_based:
            num_iter_to_print = 100
            iterable_sequence = self._movie_ids
        else:
            num_iter_to_print = 1000
            iterable_sequence = self.similarity_model.index.to_list()

        for i, movieID in enumerate(iterable_sequence):
            # Print information about execution progress and estimated completion.
            if i != 0 and i % num_iter_to_print == 0:
                remaining_movies = num_movies - i
                elapsed_time = time.time() - start_time
                print(f"Remaining movie predictions for user {userID}: {remaining_movies}")
                print(f"Computing time for {num_iter_to_print} movies: {elapsed_time:.2f} seconds")
                total_elapsed_time += elapsed_time
                average_time_per_movie = total_elapsed_time / i
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
                score = self.predict_score(userID, movieID)
                tuples.append((int(movieID), title, float(score)))

        print("Predictions computed.")

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        tuples.sort(key=lambda x: (-x[2], x[0]))

        # Save the state of the machine
        self._current_user = userID
        self._predictions = tuples

        # Print and return the top N
        topN = tuples[:n]

        print("Done.")

        print(f"Printing top {n}..")
        printTopN(topN)

        return topN

    def count_positive_elements_model(self):
        """
            Count the number of strictly positive elements in the similarity model.

            Returns:
            - int: Number of strictly positive elements.
        """
        # Create a boolean mask for positive values
        positive_mask = self.similarity_model > 0

        # Count the number of strictly positive elements
        positive_count = positive_mask.values.sum()

        return positive_count

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
