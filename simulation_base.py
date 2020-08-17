import numpy as np
import matplotlib.pyplot as plt


#======================================
# set up hypothesis/concept space
#======================================
def create_boundary_hyp_space(n_features):
    """
    Creates a hypothesis space of concepts defined by a linear boundary.
    Each row is a hypothesis. Print it to see what it looks like.
    """
    hyp_space = []
    for i in range(n_features + 1):
        hyp = [1 for _ in range(n_features)]
        hyp[n_features-i:n_features] = [0 for _ in range(i)]
        hyp_space.append(hyp)
    hyp_space = np.array(hyp_space)
    return hyp_space


def create_line_hyp_space(n_features):
    """
    Creates a hypothesis space of concepts defined by 1D lines.
    Each row is a hypothesis. Print it to see what it looks like.
    """
    hyp_space = []
    for i in range(1, n_features + 1):
        for j in range(n_features - i + 1):
            hyp = [0 for _ in range(n_features)]
            hyp[j:j + i] = [1 for _ in range(i)]
            hyp_space.append(hyp)
    hyp_space = np.array(hyp_space)
    return hyp_space


#======================================
# define active learner
#======================================
# core components: Bayes update, expected information gain
def init_prior(n_hyp, n_features, n_labels):
    """
    Output:
    prior
    shape: 1-d np array (n_hypothesis)
    meaning: prior[i] is the prior probability for hypothesis i being
        the correct hypothesis, or P(h=i).
    """
    prior = 1/n_hyp*np.ones(n_hyp)
    return prior


def likelihood(n_features, n_labels, hyp_space):
    """
    Input:
    n_features: number of n_features
    n_labels: number of n_labels
    hyp_space: from create_boundary_hyp_space(n_features)
    Output:
    lik
    shape: 3-d np array (n_hypothesis, n_features, n_labels)
    meaning: given that hypothsis i is true, then the probability that
        feature j takes on label k is lik[i,j,k], or P(y=k|x=j,h=i).
    """
    n_hyp = len(space)
    lik = np.zeros((n_hyp, n_features, n_labels))
    for i, hyp in enumerate(hyp_space):
        for j, feature in enumerate(range(n_features)):
            for k, label in enumerate(range(n_labels)):
                if hyp[feature] == label:
                    lik[i, j, k] = 1
                else:
                    lik[i, j, k] = 0
    return lik


def posterior(xs, ys, prior, lik):
    """
    Input:
    xs: a list of features probed
    ys: a list of corresponding labels
    like: the 3-d array from likelihood(...)
    prior: the 1-d array from init_prior(...) or
        previous posterior from posterior(...)
    Output:
    post
    shape: 1-d np array (n_hyp)
    meaning: post[i] is the posterior probability of hypothesis i given
        xs, ys and the prior, or P(h=i|xs,ys).
    """
    # Bayes' Rule: P(h|xs,ys) = P(ys|xs,h)P(h) / [sum_h P(ys|xs,h)P(h)].
    # P(h) is the prior; P(y|x,h) is the lik.
    # Will want to account for cases impossible scenarios where
    # (numerator = 0 everywhere)due to imageined selection in the
    # computation of expected information gain.
    n_hyp = len(prior)
    post = np.zeros(n_hyp)
    # fill in details
    return post

def predictive(prior, lik):
    """
    Input:
    prior: the 1-d array from init_prior(...) or
        previous posterior from posterior(...)
    lik: the 3-d array from likelihood(...)
    Output:
    pred
    shape: 2-d np array (n_features, n_labels)
    meaning: pred[j,k] is the predictive probability that feature j has
        label k under the prior, or P(y=k|x=j).
    """
    # P(y|x) = sum_h p(y|x,h)P(h) for prior predictive
    # P(y|x) = sum_h p(y|x,h)P(h|D) for posterior predictive
    n_hyp, n_features, n_labels = lik.shape
    pred = np.zeros([n_features, n_labels])
    # fill in details
    return pred


def entropy(prob_vec):
    """
    Input:
    prob_vec: a 1-d array that sums to 1.
    Output:
    The entropy of prob_vec.
    Returns 0 if prob_vec is all 0s.
    Note:
    Another convention is to use the negative of this.
    """
    non_zero_vals = prob_vec[prob_vec!=0]
    return np.sum(non_zero_vals*np.log(non_zero_vals))


def normalize(vec):
    return vec/np.sum(vec)


def expected_information_gain(prior, lik):
    """
    Input:
    prior: the 1-d array from init_prior(...) or
        previous posterior from posterior(...)
    lik: the 3-d array from likelihood(...)
    Output:
    eig
    shape: 1-d np array (n_features)
    """
    # EIG(x') = [sum_y P(y|x')H(h|y,x')] - H(h)
    # H(a) is the entropy of P(a), so H(h) is entropy(prior),
    # and H(h|y,x') is entropy(post)
    # fill in details
    return eig


#======================================
# a test case
#======================================
# define hypothesis space
n_labels = 2
n_features = 3
space = create_boundary_hyp_space(n_features)
print(space)
# Expected output:
# [[1 1 1]
#  [1 1 0]
#  [1 0 0]
#  [0 0 0]]
# Each row is a hypothesis, h.
# Each column is a feature, x.
# The 0 and 1 are labels, y.
# Think of this as a boundary search task. There are three buttons.
# You earn $1 by pressing buttons to the left of the boundary and $0 otherwise.
# You want to know: where is the boundary?

# get prior and likelihood
n_hyp = len(space)
prior = init_prior(n_hyp, n_features, n_labels)
lik = likelihood(n_features, n_labels, space)

# get posterior
xs = [1]
ys = [1]
post = posterior(xs, ys, prior, lik)
print(post)
# Expected output:
# [0.5 0.5 0.  0. ]
# This is the posterior probability that each hypothesis is true given data xs, ys
# Does this make sense?
# The first entry says that hypothesis 1---[1,1,1]---has 1/2 probability
# of being the underlying hypothesis.
# We can see that only the first two hypotheses are consistent with x=1,y=1,
# so this makes sense.

# get predictive
prior_pred = predictive(prior, lik)
print(prior_pred)
# Expected output:
# [[0.25 0.75]
#  [0.5  0.5 ]
#  [0.75 0.25]]
# This is the  prior predictive probability that feature i has label k under the prior.
# The row indexes feature, and the column indexes label.
# The (1,1) entry is the probability that feature 1 has label 0.
# From the hypothesis space, we can tell that the chance of tihs happening is
# indeed 1/4 if each hypothesis is uniformly likely to occur.

# eget xpected information gain
eig = expected_information_gain(prior, lik)
print(eig)
# Expected output:
# [0.56233514 0.69314718 0.56233514]
# This is the expected information gain, or equivalently, the reduction of
# uncertainty about which hypothesis is the right one, after probing feature x.
# The output shows that probing the middle feature reduces the uncertainty the most.


#======================================
# active-learning loop
#======================================
# define hypothsis space
n_labels = 2
n_features = 11
space = create_boundary_hyp_space(n_features)

# get prior and likelihood
n_hyp = len(space)
prior = init_prior(n_hyp, n_features, n_labels)
lik = likelihood(n_features, n_labels, space)

# assign underlying true hypothesis
# fill in details

print("hypothesis space:")
print(space)

print("selected hypothesis:")
print(hyp)

print("Current belief:")
print(prior)

# the active-learning loop
# pre-set the number of active-learning steps
n_step = 4
for step in range(n_step):
    # fill in details
    # calculate expected information gain
    # select the feature x with the highest EIG
    # get the label y from the underlying true hypothesis
    # calculate the posterior
    # update the prior to the posterior


# If n_features = 11, and hyp = space[0], the sequence of x is 5,8,9,10
# If n_features = 11, and hyp = space[-1], the sequence of x is 5,2,0,1
# The asymetry of the last two selections from the two sequences should be
# an artifact of the argmax always choosing the smaller ind if two inds have
# equal values.
