__author__ = "Dario Mameli"

from UUCFRecommender import *
from IICFRecommender import *
from collections import defaultdict


class HybridRecommender:
    """
        Hybrid Recommender System combining UUCF and IICF strategies.
    """
    def __init__(self, *args):
        """
            Initialize the HybridRecommender.

            Parameters:
            - args: Tuple of arguments. It can either contain two strings representing the paths to the movies dataset
              and the ratings dataset, or two instances of UUCFRecommender and IICFRecommender.

            Raises:
            - NotImplementedError: If the arguments provided are not valid.
        """
        print("------------"
              "HYBRID RECOMMENDER"
              "------------")
        # If strings are provided then we want to build the recommenders from scratch
        if isinstance(args[0], str) and isinstance(args[1], str):
            pathToMoviesDataset = args[0]
            pathToRatingsDataset = args[1]
            self._uucf = UUCFRecommender(pathToMoviesDataset, pathToRatingsDataset)
            self._iicf = IICFRecommender(pathToMoviesDataset, pathToRatingsDataset)
        # Else if the recommenders are provided they are saved at their current state
        elif isinstance(args[0], UUCFRecommender) and isinstance(args[1], IICFRecommender):
            self._uucf = args[0]
            print("Loaded UUFCRecommender")
            self._iicf = args[1]
            print("Loaded IICFRecommender")
        elif isinstance(args[1], UUCFRecommender) and isinstance(args[0], IICFRecommender):
            self._uucf = args[1]
            print("Loaded UUCFRecommender")
            self._iicf = args[0]
            print("Loaded IICFRecommender")

        else:
            raise NotImplementedError("Can only provide paths to the datasets or the already instantiated recommenders!"
                                      )

    def topN(self, userID, n=10, weight_uucf=0.5, weight_iicf=0.5):
        """
            Compute the top N recommendations for a given user using hybrid recommendation approach.

            Parameters:
            - userID: The ID of the user for whom recommendations are to be generated.
            - n: The number of recommendations to be generated.
            - weight_uucf: Weight assigned to the UUCF predictions.
            - weight_iicf: Weight assigned to the IICF predictions.

            Returns:
            - topN_weighted: A list of tuples containing the top N recommendations, each tuple containing
              (movieID, title, predicted_score).
        """
        print(f"Computing HYBRID top {n} for user {userID}..")

        print(f"Computing all predictions with UUCF")
        uucf_predictions = self._uucf.get_predictions(userID, n)
        print("Done.")

        print(f"Computing all predictions with IICF")
        iicf_predictions = self._iicf.get_predictions(userID, n)
        print("Done.")

        # Where to save all predictions with weighted scores
        predictions_weighted = []

        for tuple_ in uucf_predictions:
            new_tuple = tuple_[:2] + (tuple_[2] * weight_uucf,)
            predictions_weighted.append(new_tuple)

        for tuple_ in iicf_predictions:
            new_tuple = tuple_[:2] + (tuple_[2] * weight_iicf,)
            predictions_weighted.append(new_tuple)

        # Create a dictionary to accumulate predictions
        prediction_sums = defaultdict(float)

        # Sum predictions for each user-item pair
        for movie_id, title, prediction in predictions_weighted:
            prediction_sums[(movie_id, title)] += prediction

        # Convert dictionary back to a list of tuples
        predictions_weighted = [(movie_id, title, prediction_sum) for (movie_id, title), prediction_sum in
                                prediction_sums.items()]

        # Sort by movie_scores in descending order of score and in case of ties by MovieID in ascending order
        predictions_weighted.sort(key=lambda x: (-x[2], x[0]))

        # Print and return the top n
        topN_weighted = predictions_weighted[:n]

        print(f"Hybrid top {n} computed.")

        print(f"Printing top {n}..")
        printTopN(topN_weighted)

        return topN_weighted

    def retrieve_rating_history(self, userID, max_num=np.inf):
        """
            Retrieve the rating history of a user.

            Parameters:
            - userID: The ID of the user whose rating history is to be retrieved.
            - max_num: Maximum number of ratings to retrieve.

            Returns:
            - None: simply prints the rating history by calling the function of the uucf recommender.
        """
        return self._uucf.retrieve_rating_history(userID, max_num)
