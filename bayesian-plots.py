"""
This file contains the python code to generate the plots in my post on
"Bayesian Fundamentals" at the following link:

https://mml.johnmyersmath.com/probabilistic%20programming/2022/12/28/pp1.html

name: john myers
date: dec 30, 2022
"""

import numpy as np
from scipy.stats import beta, binom, uniform
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
The following code produces the plot titled "full (simulated)
joint distribution."
"""

# Get a uniform sample of evenly spaced p-values on the interval [0, 1].
# Define an emtpy list to hold the (x, p) values.
p_sample = np.arange(0, 1.01, 0.01)
data = []

# For each p value, choose 100 from draws from the distribution of x
# with that p value.
n = 100
for p in p_sample:
    for k in range(n):
        x = binom.rvs(n=n, p=p)
        data.append((p, x))

# Define a dataframe containing the (x, p) values.
df = pd.DataFrame(data, columns=['p', 'x'])

# The scatter plot and marginal plots will be contained in a seaborn
# JointGrid object.
g = sns.JointGrid()

# Add the plots onto the JointGrid object. Note the `bins=1` parameter setting
# for the marginal plot of the p values.
sns.scatterplot(data=df, x='x', y='p', alpha=0.5, linewidth=0, ax=g.ax_joint)
sns.histplot(data=df, x='x', ax=g.ax_marg_x, fill=False, linewidth=1.5, bins=50, kde=True)
sns.histplot(data=df, y='p', ax=g.ax_marg_y, bins=1, fill=False, linewidth=1.5)

# Set titles and print.
g.set_axis_labels(xlabel=r'$x$ values', ylabel=r'$p$ values')
plt.suptitle('Full (simulated) joint distribution')
plt.tight_layout()

"""
The following code produces the plot titled "a pair of (simulated) conditional
distributions."
"""

# Slice the dataframe to grab only those x values with corresponding
# p values of either 0.4 or 0.8.
df_slice = df.loc[(df['p'] == 0.4) | (df['p'] == 0.8), :]

# Again, the plots will land on a JointGrid object from seaborn.
g = sns.JointGrid()

# Plot a scatter plot for the data with p values *not* equal to 0.4 or 0.8.
# Dim the alpha value,
sns.scatterplot(data=df, x='x', y='p', alpha=0.05, linewidth=0, ax=g.ax_joint)

# Plot a scatter plot for the data with p values equal to 0.4 or 0.8.
sns.scatterplot(data=df_slice, x='x', y='p', alpha=0.5, linewidth=0, ax=g.ax_joint)

# Plot the conditional distributions for the x values.
sns.histplot(data=df_slice, x='x', ax=g.ax_marg_x, fill=False, linewidth=1.5, bins=50)

# Draw a "histogram" by hand (really, a line plot) to represent the p values
# of 0.4 and 0.8.
sns.lineplot(x=[0, 1], y=[0.4, 0.4], ax=g.ax_marg_y, linewidth=1.5)
sns.lineplot(x=[0, 1], y=[0.8, 0.8], ax=g.ax_marg_y, linewidth=1.5)

# Set titles and print.
g.set_axis_labels(xlabel=r'$x$ values', ylabel=r'$p$ values')
plt.suptitle('A pair of (simulated) conditional distributions')
plt.tight_layout()

"""
The following code produces the plots titled "prior vs posterior distribution."
"""

# Suppose that we've observed the ball roll to the *left* 42 times...
x_observed = 42

# The prior distribution for p is uniform, while the posterior is a
# beta distribution. Sample from both of them.
prior = uniform.pdf(p_sample)
posterior = beta.pdf(p_sample, a=x_observed + 1, b=n - x_observed + 1)

# Draw a simple matplotlib line plot.
plt.plot(p_sample, prior, label=r'uniform prior')
plt.plot(p_sample, posterior, label=r'posterior with $(n,x)=(100, 42)$')
plt.legend()
plt.xlabel(r'$p$ values')
plt.ylabel('density')
plt.suptitle(r'prior vs. posterior distribution')
plt.tight_layout()

"""
The following code produces the ridgeline plot titled "progression through ten updates
to the posterior."
"""

# The true p value is 0.8, and we've observed n=10 data points (i.e., x values)
# from a binomial distribution.
p = 0.8
n = 10
x_observed = binom.rvs(size=n, p=p, n=1, random_state=42)

# Generate a list that will provide the labels along the y-axis of our plot that
# looks like: 1 sample, 2 samples, 3 samples, etc.
num_samples = ['1 sample']
for k in range(2, n + 1):
    num_samples.append(str(k) + ' samples')


# Step along the observed data in x_observed, and compute the posterior beta
# distributions. For each level of observations (e.g., 1 observation, 2 observations,
# etc.) there will be a corresponding plot on the ridgeline plot. Each of these subplots
# will come from its own dataframe, so we need to initialize an empty list of
# dataframes.
df_list = []
for k in range(n):
    # Sum the x observations up to the k-th observation.
    x_sum = sum(x_observed[: k + 1])
    # For those x observations, sample from the posterior density.
    density = beta.pdf(p_sample, a=x_sum + 1, b=k + 1 - x_sum + 1)
    # Put these densities into a 2-column dataframe.
    df = pd.DataFrame({'p': p_sample, 'density': density})
    # Put a third column into the dataframe indicating the number of observations.
    labels = pd.Series(np.tile(num_samples[k], len(density)), name='label')
    # Add this dataframe to the list of dataframes.
    df_list.append(pd.concat([df, labels], axis=1))

# Stack all the individual dataframes on top of each other to create a big
# 3-column dataframe. We need to also reset the indices for some annoying reason.
df = pd.concat(df_list).reset_index()

# The ridgeline plot will be created using a seaborn relplot.
g = sns.relplot(data=df,
                x='p',
                y='density',
                hue='label',
                kind='line',
                row='label',
                aspect=5,
                legend=False,
                height=1,
                facet_kws={'sharey': False},
                )

# Alter the titles.
g.set_titles('')
g.set(yticks=[], ylabel="", xlabel=r'$p$ values')
g.despine(bottom=True, left=True)

# Give custom labels to each of the individual subplots.
for k in range(10):
    g.axes[k][0].set_ylabel(str(k + 1) + ' obs.')

# Set more titles and print.
plt.suptitle('Progression through ten updates to the posterior')
plt.tight_layout()
