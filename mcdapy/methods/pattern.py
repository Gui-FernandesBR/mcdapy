__author__ = "Guilherme Fernandes Alves"
__email__ = "gf10.alves@gmail.com"

import matplotlib.pyplot as plt
import numpy as np


class Pattern:
    """The PATTERN method. Simple and fast method for multi-criteria decision
    analysis. It is based on the concept of pertinence index. The method is
    described in ... .

    Attributes
    ----------
    Pattern.alternatives : tuple
        The names of the alternatives.
    Pattern.criteria : tuple
        The names of the criteria.
    Pattern.matrix : np.array
        The array of the alternatives and criteria. Must contain only positive
        values. The number of rows must be equal to the number of alternatives
        and the number of columns must be equal to the number of criteria.
        Example: np.array([[1, 2, 3], [4, 5, 6]]) has 2 alternatives and 3
        criteria.
    Pattern.weights : np.array
        The array of the weights of the criteria. Must contain only a single
        row. Values must be positive and can not sum up to 0.
    Pattern.n_alternatives : int
        The number of alternatives.
    Pattern.n_criteria : int
        The number of criteria.
    Pattern.weighted_matrix : np.array
        The weighted matrix of the alternatives and criteria.
    Pattern.pertinence_index : np.array
        The array of the pertinence indexes of the alternatives.
    Pattern.rankings : np.array
        The array of the rankings of the alternatives.

    Public Methods
    -------
    Pattern.solve():
        Run the PATTERN method.
    Pattern.print_rankings():
        Print the rankings of the alternatives.
    Pattern.plot_rankings():
        Plot the rankings of the alternatives.
    Pattern.plot_matrix():
        Plot the matrix of the alternatives and criteria.

    Example
    -------
    >>> import numpy as np
    >>> from mcdapy.methods import Pattern
    >>>
    >>> # Define the alternatives, criteria and matrix
    >>> alternatives = ("A", "B", "C")
    >>> criteria = ("C1", "C2", "C3")
    >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>>
    >>> # Define the weights
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>>
    >>> # Initialize the Pattern class
    >>> pattern = Pattern(alternatives, criteria, matrix, weights)
    >>>
    >>> # Run the PATTERN method
    >>> pattern.solve()
    >>>
    >>> # Print the results
    >>> print(pattern.pertinence_index)
    >>> print(pattern.rankings)
    """

    def __init__(
        self,
        alternatives: tuple,
        criteria: tuple,
        matrix: np.array,
        weights: np.array,
    ) -> None:
        """Initialize the Pattern class

        Parameters
        ----------
        alternatives : tuple
            The names of the alternatives.
        criteria : tuple
            The names of the criteria.
        matrix : np.array
            The array of the alternatives and criteria. Must contain only positive
            values. The number of rows must be equal to the number of alternatives
            and the number of columns must be equal to the number of criteria.
            Example: np.array([[1, 2, 3], [4, 5, 6]]) has 2 alternatives and 3
            criteria.
        weights : np.array
            The array of the weights of the criteria. Must contain only a single
            row. Values must be positive and can not sum up to 0.

        Returns
        -------
        None
        """

        # Store the inputted information
        self.alternatives = alternatives
        self.criteria = criteria
        self.matrix = matrix
        self.weights = weights

        # Check if the inputs are valid
        self.__check_inputs()

        # Process useful information
        self.n_alternatives = len(alternatives)
        self.n_criteria = len(criteria)

        return None

    # Public methods

    def solve(self) -> None:
        """Solve the system based on the PATTERN method.

        Returns
        -------
        None
        """
        # Initialize variables
        pertinence_index = np.zeros(self.n_alternatives)
        weighted_matrix = np.zeros((self.n_alternatives, self.n_criteria))

        # Step 1: Check if the weights sum up to 1
        self.__normalize_weights()

        # Step 2: Check if the matrix is normalized
        self.__normalize_matrix()

        # Step 3: Calculate the weighted sum of each alternative
        self.__calculate_weighted_matrix(weighted_matrix)
        self.__calculate_pertinence_indexes(pertinence_index)

        # Step 4: Calculate the ranking of the alternatives
        self.__calculate_rankings()

        return None

    # Private methods

    def __check_inputs(self) -> None:
        """Check if all the inputs are valid, based on different criteria.
        ValueErrors are raised if any problem is found, and the program is
        terminated.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Whether any of the inputs are invalid.
        """

        # Check if all the weights are positive
        if not all(self.weights >= 0):
            raise ValueError("All the weights must be positive!")

        # Check if all the values of the matrix are positive
        if not (self.matrix >= 0).all():
            raise ValueError("All the values of the matrix must be positive!")

        # Check if len of weights is equal to len of criteria
        if len(self.weights) != len(self.criteria):
            raise ValueError(
                "The number of weights must be equal to the number of criteria!"
            )

        # Check if the weights do not sum up to 0
        if np.sum(self.weights) < 1e-10:
            raise ValueError("The sum of the weights must not be 0!")

        # Check if the len of matrix is equal to len of alternatives
        if len(self.matrix) != len(self.alternatives):
            raise ValueError(
                "The number of alternatives must be equal to the number of rows of the matrix!"
            )

        # Check if the len of each row of matrix is equal to len of criteria
        if not all(
            len(self.matrix[i]) == len(self.criteria) for i in range(len(self.matrix))
        ):
            raise ValueError(
                "The number of criteria must be equal to the number of columns of the matrix!"
            )

        return None

    def __calculate_rankings(self) -> None:
        """Calculate the ranking of the alternatives based on the pertinence
        index. The results are stored as class attributes

        Returns
        -------
        None
        """
        # Sort the alternatives by their pertinence index
        self.ranking = np.argsort(self.pertinence_index)[::-1]
        # Store the names of the alternatives in the ranking
        self.ranking_named = [self.alternatives[i] for i in self.ranking]
        return None

    def __calculate_pertinence_indexes(self, pertinence_index: np.array) -> None:
        """Calculate the pertinence index of each alternative based on the
        weighted matrix. The results are stored as class attributes.
        The pertinence index of an alternative is the sum of the weighted
        values of this alternative with regard to all the criteria.

        Parameters
        ----------
        pertinence_index : np.array
            The initialized array of the pertinence index of each alternative.
            Usually contains only zeros.

        Returns
        -------
        None
        """
        for i in range(self.n_alternatives):
            pertinence_index[i] += np.sum(self.weighted_matrix[i, :])
        self.pertinence_index = pertinence_index
        return None

    def __calculate_weighted_matrix(self, weighted_matrix: np.array) -> None:
        """Calculate the weighted matrix based on the main matrix and the weights.
        The results are stored as class attributes.

        Parameters
        ----------
        weighted_matrix : np.array
            The initialized array of the weighted matrix. Usually contains only
            zeros.

        Returns
        -------
        None
        """
        for i in range(self.n_alternatives):
            for j in range(self.n_criteria):
                weighted_matrix[i, j] = (
                    self.normalized_matrix[i, j] * self.normalized_weights[j]
                )
        self.weighted_matrix = weighted_matrix
        return None

    def __normalize_matrix(self):
        """Normalize the matrix based on the sum of each column. The results
        are stored as class attributes. A normalized column is a column that
        sums up to 1. And a normalized matrix is a matrix that has all its
        columns normalized.

        Returns
        -------
        None
        """

        self.normalized_matrix = self.matrix.copy()

        for i in range(self.n_criteria):
            if np.sum(self.matrix[:, i]) - 1 < 1e-10:
                # This column is already normalized
                pass
            else:
                # Normalize this specific column
                self.normalized_matrix[:, i] = self.matrix[:, i] / np.sum(
                    self.matrix[:, i]
                )

        return None

    def __normalize_weights(self) -> None:
        """Normalize the weights based on the sum of all the weights. The
        results are stored as class attributes. The weights are considered
        normalized if they sum up to 1. It is worth to mention that the weights
        are always non-negative.

        Returns
        -------
        None
        """

        self.normalized_weights = self.weights.copy()
        if np.sum(self.weights) - 1 < 1e-10:
            # The weights are already summing 1
            pass
        else:
            # Normalize the weights
            self.normalized_weights = self.weights / np.sum(self.weights)

        return None

    # Post-processing methods

    def print_rankings(self) -> None:
        """Print the rankings of the alternatives, the easiest way to see the
        results of the Pattern method.

        Returns
        -------
        None
        """
        try:
            print("The ranking of the alternatives is: {}".format(self.ranking))
            print("The ranking of the alternatives is: {}".format(self.ranking_named))
        except AttributeError:
            print("It seems that the rankings have not been calculated yet!")
            print("Please, run the solve() method first!")

        return None

    def plot_matrix(self, return_fig=False):
        # Hey, I will document this later!

        # Set the labels
        labels = [i for i in self.alternatives]
        criteria = [i for i in self.criteria]

        # Set the colors
        colors = plt.cm.Blues(np.linspace(0, 0.5, self.n_alternatives))

        # Plot the matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(self.weighted_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Set the ticks
        ax.set_xticks(np.arange(self.n_criteria))
        ax.set_yticks(np.arange(self.n_alternatives))
        ax.set_xticklabels(criteria)
        ax.set_yticklabels(labels)

        # Rotate the labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

        # Set the title
        ax.set_title("Normalized weighted matrix")

        # Set the values
        for i in range(self.n_alternatives):
            for j in range(self.n_criteria):
                text = ax.text(
                    j,
                    i,
                    round(self.weighted_matrix[i, j], 2),
                    ha="center",
                    va="center",
                    color="black",
                )

        if not return_fig:
            # Show the plot
            plt.show()
            return None

        return fig, ax

    def plot_rank(self, return_fig=False) -> None:
        # Hey, I will document this later!

        # Set the colors
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, self.n_alternatives))

        # Sort the labels and the pertinence index
        rank = self.ranking.tolist()
        rank.reverse()
        labels = [i for i in self.ranking_named]
        labels.reverse()
        pertinence_index = [self.pertinence_index[i] for i in rank]

        # Plot the ranking
        fig, ax = plt.subplots()
        ax.barh(labels, pertinence_index, color=colors)

        # Set the title
        ax.set_title("Pertinence index of each alternative")

        # Set the values
        for i, v in enumerate(pertinence_index):
            ax.text(v, i, "{:.2f}".format(v), va="center", color="black", ha="right")

        if not return_fig:
            # Show the plot
            plt.show()
            return None

        return fig, ax
