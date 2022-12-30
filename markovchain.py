import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MarkovChain:
    """
    A class used to simulate and plot the trajectory of a Markov chain with state space {1,2,...c}, where c is a
    user-defined variable.

    Attributes
    -----------
    transition_matrix : numpy array of shape (c, c)
        The transition matrix of the chain, where `c` is the size of the state space. This attribute is set to
        None if `compute_trajectory` has not been called beforehand.
    state_space : numpy array of shape (c,)
        The state space of the chain, where `c` is the size of the state space. This attribute is set to None
        if `compute_trajectory` has not been called beforehand.
    trajectory : list of int's
        The trajectory of the chain. This attribute is set to None if `compute_trajectory` has not been called
        beforehand.

    Methods
    -----------
    compute_trajectory(self, transition_matrix, n_iter=100, initial_state=1)
        Compute the trajectory of the Markov chain.
    trace_plot(self, width_ratios=[8, 1], figsize=(8, 3))
        Plots the trace of the trajectory, along with a bar plot showing the density (i.e., frequency) that each
        state is visited.
    """
    def __init__(self):
        self.transition_matrix = None
        self.state_space = None
        self.trajectory = None
        self._initial_state = None
        self._state_space_size = None
        self._n_iter = None

    def compute_trajectory(self, transition_matrix, n_iter=100, initial_state=1):
        """
        Computes the trajectory of the Markov chain.

        :param transition_matrix: transition matrix
        :param n_iter: length of trajectory
        :param initial_state: initial state in which the trajectory begins
        :return: None
        """
        self._state_space_size = transition_matrix.shape[0]
        self.transition_matrix = transition_matrix
        self._n_iter = n_iter
        self._initial_state = initial_state

        # Compute the trajectory of the chain, beginning with the initial state.
        self.trajectory = [self._initial_state]
        current_state = initial_state
        self.state_space = np.arange(1, self._state_space_size + 1)
        for i in range(self._n_iter):
            # Choose the next state randomly, with probabilities given by the relevant rows of the
            # transition matrix.
            next_state = np.random.choice(self.state_space, p=self.transition_matrix[current_state - 1, :])
            self.trajectory.append(next_state)
            current_state = next_state

    def trace_plot(self, width_ratios=[8, 1], figsize=(8, 3)):
        """
        Displays a trace plot of the Markov chain. The `compute_trajectory` method must
        be called beforehand.

        :param width_ratios: list of length 2 giving the ratio of widths of the trace plot to the distribution plot
        :param figsize: tuple of length 2 representing the width and height of the figure (in inches)
        :return: None
        """
        if self.trajectory is None:
            print('Run the get_trajectory() method first!')
            return

        # Define a dataframe with two columns: the first column counts the number of steps in the trajectory
        # (to be displayed along the horizontal axis of the trace plot) while the second column is the
        # trajectory itself.
        steps = np.arange(0, self._n_iter + 1)
        df_trajectory = pd.DataFrame({'n': steps, 'state': self.trajectory})

        # Compute the densities (i.e., the frequency of visits) of the trajectory. These numbers will be displayed
        # as a bar plot to the right of the trace plot.
        density_dict = {}
        for i in range(self._state_space_size):
            prop = len(df_trajectory[df_trajectory['state'] == i + 1]) / self._n_iter
            density_dict = density_dict | {i + 1: prop}

        # Initialize the matplotlib figure and axes objects.
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, width_ratios=width_ratios, figsize=figsize)

        # The trace plot.
        ax1.plot(df_trajectory['n'], df_trajectory['state'], 'o-', markersize=2)
        ax1.set_yticks(list(range(1, self._state_space_size + 1)))
        ax1.set_xlabel(r'$n$')
        ax1.set_ylim(0.8, self._state_space_size + 0.2)
        ax1.set_ylabel('state')
        ax1.set_xlim(-1, self._n_iter)

        # Add a bar for each state in the bar plot.
        for i in range(self._state_space_size):
            ax2.plot([0, density_dict[i + 1]],
                     [i + 1, i + 1],
                     linewidth=6.5,
                     solid_capstyle='butt')

        ax2.set_xlabel('density')

        fig.suptitle(
            r'trace plot for ${}$-state markov chain with $n={}$'.format(self._state_space_size, self._n_iter))
        plt.tight_layout()
