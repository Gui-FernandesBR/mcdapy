__author__ = "Guilherme Fernandes Alves"
__email__ = "gf10.alves@gmail.com"

import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from .methods import Pattern


class MonteCarlo:
    def __init__(self, input_dictionary: dict, n: int, method: str = "PATTERN") -> None:
        self.input_dictionary = input_dictionary
        self.method = method
        self.n = n
        self.distribution_map = {
            "normal": np.random.normal,
            "uniform": np.random.uniform,
            "lognormal": np.random.lognormal,
            "binomial": np.random.binomial,
            "poisson": np.random.poisson,
            "exponential": np.random.exponential,
            "gamma": np.random.gamma,
            "beta": np.random.beta,
        }

        self.__define_arrays_from_dict()
        self.__create_weight_sets()
        self.__create_matrix_sets()
        self.__create_normalized_matrix_sets()
        self.pertinence_index = {i: {} for i in self.alternatives}
        self.rank_results = copy.deepcopy(self.pertinence_index)
        self.normalized_matrix = copy.deepcopy(self.pertinence_index)
        self.found_ranking = False
        self.found_confusion_matrix = False

        return None

    def __define_arrays_from_dict(self) -> None:

        self.alternatives = list(self.input_dictionary["alternatives"].keys())
        self.criteria = list(
            self.input_dictionary["alternatives"][self.alternatives[0]].keys()
        )
        self.n_alternatives = len(self.alternatives)
        self.n_criteria = len(self.criteria)

        self.weights = np.array(
            [self.input_dictionary["weights"][i][0] for i in range(self.n_criteria)]
        )
        self.matrix = np.array(
            [
                [
                    self.input_dictionary["alternatives"][alternative][criterion][0]
                    for criterion in self.criteria
                ]
                for alternative in self.alternatives
            ]
        )

        self.stdev_weights = np.array(
            [self.input_dictionary["weights"][i][1] for i in range(self.n_criteria)]
        )

        self.distribution_weights = list(
            [self.input_dictionary["weights"][i][2] for i in range(self.n_criteria)]
        )

        self.stdev_matrix = np.array(
            [
                [
                    self.input_dictionary["alternatives"][alternative][criterion][1]
                    for criterion in self.criteria
                ]
                for alternative in self.alternatives
            ]
        )

        self.distributions_matrix = list(
            [
                [
                    self.input_dictionary["alternatives"][alternative][criterion][2]
                    for criterion in self.criteria
                ]
                for alternative in self.alternatives
            ]
        )

        return None

    def __create_weight_sets(self) -> None:
        # Generate a set of weights with the corresponding distribution
        distribution = [self.distribution_map[d] for d in self.distribution_weights]
        weight_sets = {}
        for i in range(self.n):
            weight_sets[i] = [
                distribution[i](self.weights[i], self.stdev_weights[i])
                for i in range(self.n_criteria)
            ]
        self.weight_sets = weight_sets
        return None

    def __create_matrix_sets(self):
        distribution = [
            [self.distribution_map[d] for d in self.distributions_matrix[i]]
            for i in range(len(self.distributions_matrix))
        ]
        matrix_sets = {}
        for i in range(self.n):
            for j in range(self.n_alternatives):
                matrix_sets[i] = [
                    [
                        distribution[i][j](self.matrix[i][j], self.stdev_matrix[i][j])
                        for j in range(len(self.matrix[i]))
                    ]
                    for i in range(len(self.matrix))
                ]

        # Remove the negative values from the matrix sets
        # This is a temporary fix for the negative values that are generated

        for i in range(len(matrix_sets)):
            for j in range(len(matrix_sets[i])):
                for k in range(len(matrix_sets[i][j])):
                    if matrix_sets[i][j][k] < 0:
                        matrix_sets[i][j][k] = 0

        self.matrix_sets = matrix_sets

        return None

    def __create_normalized_matrix_sets(self) -> None:
        normalized_matrix_sets = {}
        for i in range(self.n):
            normalized_matrix_sets[i] = [
                [
                    self.matrix_sets[i][j][k] / sum(self.matrix_sets[i][j])
                    for k in range(len(self.matrix_sets[i][j]))
                ]
                for j in range(len(self.matrix_sets[i]))
            ]
        self.normalized_matrix_sets = normalized_matrix_sets
        return None

    def run(self) -> None:
        for i in range(self.n):
            # Get a set of values to simulate

            sim_weights = np.array(self.weight_sets[i])
            sim_matrix = np.array(self.matrix_sets[i])

            # Run the simulation

            sim = Pattern(
                alternatives=self.alternatives,
                criteria=self.criteria,
                matrix=sim_matrix,
                weights=sim_weights,
            )
            sim.solve()

            # Retrieve results and store them in a dictionary
            for index, alternative in enumerate(self.alternatives):
                self.pertinence_index[alternative][i] = sim.pertinence_index[index]
                self.normalized_matrix[alternative][i] = sim.normalized_matrix[index]

        return None

    def all_results(self) -> tuple:
        return self.pertinence_index, self.weight_sets, self.matrix_sets

    def find_ranking(self) -> None:

        for i in range(self.n):
            # Find the ranking of each alternative at this simulation

            labels = list(self.alternatives)
            values = [self.pertinence_index[a][i] for a in labels]
            ranking = [x for _, x in sorted(zip(values, labels), reverse=True)]

            # Store the ranking in a dictionary
            for alternative in self.alternatives:
                self.rank_results[alternative][i] = ranking.index(alternative) + 1

        self.found_ranking = True

        return None

    def find_confusion_matrix(self) -> None:
        # Confusion Matrix of each pair of aij element in the matrix
        # Each cell of the confusion matrix is going to be the person correlation between
        # the two aij elements of the matrix

        n_rows = self.n_alternatives
        n_cols = self.n_criteria
        n_cells = int(n_rows * n_cols)

        # Create a confusion matrix
        confusion_matrix = np.zeros((n_cells, n_cells))

        for row in range(n_cells):
            for col in range(n_cells):
                # Get the aij elements of the matrix
                row1 = row // n_cols
                col1 = row % n_cols
                aij1 = [self.matrix_sets[i][row1][col1] for i in range(self.n)]
                aij2 = [
                    self.matrix_sets[i][col // n_cols][col % n_cols]
                    for i in range(self.n)
                ]
                confusion_matrix[row][col] = pearsonr(aij1, aij2)[0]

        self.confusion_matrix_cells = confusion_matrix

        self.found_confusion_matrix = True
        return None

    def all_plots(self) -> None:

        print("Plotting results...")
        self.plot_pertinence_index()
        self.plot_ranking_distributions()
        self.plot_weights_set()
        self.plot_matrix_set()

        return None

    def plot_pertinence_index(self) -> None:
        # Plot the distribution of pertinence index of each alternative

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        for alternative in self.alternatives:
            ax.hist(
                self.pertinence_index[alternative].values(),
                bins=int(self.n**0.5),
                alpha=0.8,
                label=alternative,
            )
        ax.legend(loc="right")
        ax.set_title("Distribution of pertinence index for each alternative")
        ax.set_xlabel("Pertinence index")
        ax.set_ylabel("Number of simulations")
        # ax.text(0.47, 0, f"n = {self.n}")
        plt.show()

        return None

    def plot_ranking_distributions(self):
        # Plot a confusion matrix of the results

        if not self.found_ranking:
            self.find_ranking()

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        # Create a confusion matrix
        confusion_matrix = np.zeros((self.n_alternatives, self.n_alternatives))
        for index, alternative in enumerate(self.alternatives):
            for j in range(self.n):
                confusion_matrix[index][self.rank_results[alternative][j] - 1] += 1

        # Plot the confusion matrix
        ax.imshow(confusion_matrix, cmap="Blues")
        ax.set_xticks(range(self.n_alternatives))
        ax.set_yticks(range(self.n_alternatives))
        ax.set_xticklabels(range(1, self.n_alternatives + 1))
        ax.set_yticklabels(self.alternatives)
        ax.set_title(f"Confusion matrix (n={self.n})")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Alternative")

        # add the scale bar to the plot
        color_bar = ax.figure.colorbar(ax.images[0])
        color_bar.ax.set_ylabel("Number of simulations", rotation=-90, va="bottom")

        # add data labels to the plot
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    int(confusion_matrix[i, j]),
                    ha="center",
                    va="center",
                    color="red",
                )

        plt.show()

        self.confusion_matrix = confusion_matrix

        return None

    def plot_weights_set(self):
        # Plot the distribution of input parameters (weights and matrix)

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        for weight in range(self.n_criteria):
            ax.hist(
                [self.weight_sets[i][weight] for i in range(self.n)],
                bins=int(self.n**0.5),
                alpha=0.75,
                label=f"Weight {weight}",
            )
        ax.legend(loc="upper right")
        ax.set_title("Distribution of weights parameters")
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of simulations")
        # ax.text(0.47, 0, f"n = {self.n}")
        plt.show()

        return None

    def plot_matrix_set(self):
        # Plot the distribution of matrix_sets parameters

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        for i in range(self.n_alternatives):
            for j in range(self.n_criteria):
                ax.hist(
                    [self.matrix_sets[k][i][j] for k in range(self.n)],
                    bins=int(self.n**0.5),
                    alpha=0.75,
                    label=f"a{i}{j}",
                )
        ax.legend(loc="right")
        ax.set_ylabel("Number of simulations")
        ax.set_title("Distribution of matrix parameters")

        return None

    def plot_normalized_set(self):
        # Plot the distribution of normalized_matrix_sets parameters

        fig = plt.figure(figsize=(7, 4 * int(self.n_alternatives)))

        for i in range(self.n_alternatives):
            ax = fig.add_subplot(self.n_alternatives, 1, i + 1)
            for j in range(self.n_criteria):
                ax.hist(
                    [self.normalized_matrix_sets[k][i][j] for k in range(self.n)],
                    bins=int(self.n**0.5),
                    alpha=0.75,
                    label=f"a{i}{j}",
                )
            ax.legend(loc="right")
            ax.set_ylabel("Number of simulations")
            ax.set_title("Distribution of matrix parameters")

        return None

        return None

    def plot_confusion_matrix(self):

        try:
            import seaborn as sns
        except ImportError:
            raise ImportError(
                f"seaborn is required to plot the confusion matrix, try to run `pip install seaborn` first."
            )

        if not self.found_confusion_matrix:
            self.find_confusion_matrix()

        labels = [
            f"a{i}{j}"
            for i in range(self.n_alternatives)
            for j in range(self.n_criteria)
        ]
        sns.heatmap(
            self.confusion_matrix_cells,
            annot=False,
            yticklabels=labels,
            xticklabels=labels,
            cbar=True,
            linewidth=0.3,
        )

        return None

    # def plot_stacked_rankings(self):
    # Plot the distribution of rank of each alternative
    # Disabled since the plot_ranking_distributions already provides the
    # same information
    # labels = ["A", "B", "C"]
    # firsts = [
    #     list(rank_results["A"].values()).count(1),
    #     list(rank_results["B"].values()).count(1),
    #     list(rank_results["C"].values()).count(1),
    # ]
    # seconds = [
    #     list(rank_results["A"].values()).count(2),
    #     list(rank_results["B"].values()).count(2),
    #     list(rank_results["C"].values()).count(2),
    # ]
    # thirds = [
    #     list(rank_results["A"].values()).count(3),
    #     list(rank_results["B"].values()).count(3),
    #     list(rank_results["C"].values()).count(3),
    # ]

    # fig = plt.figure(figsize=(7, 3))
    # ax = fig.add_subplot(111)
    # ax.bar(
    #     labels,
    #     thirds,
    #     alpha=1,
    #     label="3rd",
    #     color="red",
    # )
    # ax.bar(
    #     labels,
    #     seconds,
    #     alpha=1,
    #     label="2nd",
    #     bottom=thirds,
    #     color="orange",
    # )
    # ax.bar(
    #     labels,
    #     firsts,
    #     alpha=1,
    #     label="1st",
    #     bottom=[thirds[i] + seconds[i] for i in range(len(thirds))],
    #     color="green",
    # )
    # ax.legend(loc="upper right")
    # ax.set_title("Distribution of rank for each alternative")
    # plt.show()

    # return None
