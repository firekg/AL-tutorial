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
    1-d array shape: n_hypothesis
    Meaning:
    Prior probability for hypothesis i being the correct hypothesis.
    """
    prior = 1/n_hyp*np.ones(n_hyp)
    return prior


def likelihood(n_features, n_labels, hyp_space):
    """
    Output:
    3-d array shape: n_hypothesis, n_features, n_labels
    Meaning:
    If hypothsis i is true, then the probability that
    feature j takes on label k is lik[i,j,k]
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
    prior: the 1-d array from prior(...)
    Output:
    1-d array shape: n_hypothesis
    Meaning:
    post[i] is the posterior probability of hypothesis i given
    xs, ys and the prior
    """
    # Bayes' Rule: P(h|x,y) = P(y|x,h)P(h) / sum_h P(y|x,h)P(h)
    # numerator = P(y|x,h)P(h)
    n_hyp = len(prior)
    numer = np.zeros(n_hyp)
    for i in range(n_hyp):
        numer[i] = prior[i]
        for x, y in zip(xs, ys):
            numer[i] *= lik[i,x,y]
    post = numer/np.sum(numer)
    return post

def predictive(prior, lik):
    """
    Input:
    current prior can be previous posterior
    Output:
    2-d array shape: n_features, n_labels
    Meaning:
    pred[j,k] is the predictive probability that feature j has label k
    under the prior
    """
    # P(y|x*,D) = sum_h p(y|x',h)P(h|D)
    n_hyp, n_features, n_labels = lik.shape
    pred = np.zeros([n_features, n_labels])
    for j in range(n_features):
        for k in range(n_labels):
            pred[j,k] = np.sum(lik[:,j,k]*prior)
    return pred


def entropy(prob_vec):
    non_zero_vals = prob_vec[prob_vec!=0]
    return np.sum(non_zero_vals*np.log(non_zero_vals))


def normalize(vec):
    return vec/np.sum(vec)


def expected_information_gain(prior, lik):
    """
    Input note:
    current prior can be previous posterior
    Output:
    """
    n_hyp, n_features, n_labels = lik.shape
    # EIG(x') = sum_y P(y|x')H(h|y,x') - H(prior)
    # entropy of prior H(prior)
    prior_ent = entropy(prior)
    # predictive p(y|x')
    pred = predictive(prior, lik)
    post_3d = np.zeros([n_hyp, n_features, n_labels])
    for x in range(n_features):
        for y in range(n_labels):
            post_3d[:,x,y] = posterior([x], [y], prior, lik)
    weighted_post_ent = np.zeros(n_features)
    for j in range(n_features):
        weighted_post_ent[j] = 0
        for k in range(n_labels):
            weighted_post_ent[j] += pred[j,k]*entropy(post_3d[:,j,k])
    eig = weighted_post_ent - prior_ent
    return eig


#======================================
# a test case
#======================================
# Form hypothesis space
n_labels = 2
n_features = 3
space = create_boundary_hyp_space(n_features)
print(space)
# Each row is a hypothesis.
# Think of this as a boundary search task. There are three buttons.
# You earn $1 by pressing buttons to the left of the boundary and $0 otherwise.
# You want to know: where is the boundary?
# [[1 1 1]
#  [1 1 0]
#  [1 0 0]
#  [0 0 0]]

n_hyp = len(space)
prior = init_prior(n_hyp, n_features, n_labels)
lik = likelihood(n_features, n_labels, space)

# The posterior probability that hypothesis i is true given data xs, ys
xs = [1]
ys = [1]
post = posterior(xs, ys, prior, lik)
print(post)
# The answer should be:  [0.5 0.5 0.  0. ]
# Does this make sense?
# The first entry is the probability that hypothesis 1 [1,1,1] has 1/2
# probability of being the underlying hypothesis.
# We can see that only the first two hypotheses are consistent with x=1,y=1,
# so this makes sense.

prior_pred = predictive(prior, lik)
print(prior_pred)
# row indexes feature, column indexes label
# The prior predictive probability that feature i has label k under the prior
# [[0.25 0.75]
#  [0.5  0.5 ]
#  [0.75 0.25]]
# The (1,1) entry is the probability that feature 1 has label 0.
# From the hypothesis space, we can tell that the chance of tihs happening is
# indeed 1/4 if each hypothesis is uniformly likely to occur.

eig = expected_information_gain(prior, lik)
print(eig)
# This is the expected information gain, or equivalently, the reduction of
# uncertainty about which hypothesis is the right one.
# [0.56233514 0.69314718 0.56233514]
# Probing the middle one reduces the uncertainty the most =
# highest expected information gain

#======================================
# interaction
#======================================
