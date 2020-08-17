# Active Learning tutorial

In this tutorial, we will code the most common active learning model---expected information gain---in the simplest scenario---finding decision boundary in 1d discrete space. The file ```simulation_base.py``` provides a template.

The template already contains some functions (see comments in the functions for specifications):
- ```def create_boundary_hyp_space(n_features)```: generates the simplest hypothesis space.
- ```def init_prior(n_hyp, n_features, n_labels)```: initializes prior.
- ```def likelihood(n_features, n_labels, hyp_space)```: generates the likelihood.
- ```def entropy(prob_vec)```: outputs the entropy of the input probability vector.


## Exercise
Write the content of the following functions (see comments in the functions for more details):
- ```def posterior(xs, ys, prior, lik)```: This function implements Bayes' rule to calculate the posterior.
In this simple scenario, Bayes' rule is really just simple multiplications, additions, and division.
- ```def predictive(prior, lik)```: This function calculates the predictive distribution.
- ```def expected_information_gain(prior, lik)```: This function calculates the expected information gain of each feature.


After completing these functions, run the lines under the section named ```# a test case```.
(Just run ```python simulation_base.py``` in the terminal.)
The outputs printed in the terminal should match those in the comments.

After testing these functions, go on to the section named ```# active-learning loop```
and fill in the code by following the comments.

There are altogether 5 places for you to fill in the details (search for "# fill in details") in ```simulation_base.py```.

Here is an overview of the scenario and notation. The hypothesis takes the form:  
```
[[1 1 1]  
[1 1 0]  
[1 0 0]  
[0 0 0]]
```  
Each row is a hypothesis ```h```, so ```n_hyp``` = 4.  
Each column is a feature ```x```, so ```n_features``` = 3.  
There are two kinds of labels ```y```, 0 or 1, so ```n_labels``` = 2.

Let the prior be ```P(h)```, the likelihood be ```P(y|x,h)```,
then the posterior is ```P(h|x,y) = P(y|x,h)P(h) / [sum_h P(y|x,h)P(h)]```.  
The predictive distribution is ```P(y|x) = sum_h p(y|x,h)P(h)```,
where P(h) can really be the prior or the posterior.

The expected information gain is given by ```EIG(x') = [sum_y P(y|x')H(h|y,x')] - H(h)```,
where ```H(a)``` is the entropy of the probability distribution P(a).
So, ```H(h|y,x')``` is the entropy of a posterior, and ```H(h)``` is the entropy of the prior.

An active-learning involves: defining a concept space,
assigning an underlying true hypothesis,
selecting a feature to probe via an objective (expected information gain),
observe the label, compute the posterior to update the current belief,
and repeat the last three steps until a stopping criterium is met.

## Additional exercises
- **Code the expected information gain in the Bayesian active-sensing style.**
Formally,
```
BAS(x') = [sum_h P(h)H(y|x',h)] - H(y|x'),
```
where ```H(y|x',h)``` is the entropy of the likelihood,
and ```H(y|x')``` is the entropy of the the prior predictive.
Test this function as you tested ```def expected_information_gain(prior, lik)```
in the test case section. It should give the same answer.

- **Active learning on simplified game of battleship.**
Check out the function ```def create_line_hyp_space(n_features)```.
This function generates a hypothesis space where the hypotheses are all possible
unbroken line segments. Write a function that generates a hypothesis space
that contains all single rectangles.
This is akin to the game of battleship with a single ship of any size.
See if the active learning algorithm is sensible.

- **(Rough and tough.) Active learning on hierarchical concept space.**
Code a hierarchical hypothesis of the form:
```h_1 = [p_1', p_2', ...]; h_2 = [p_1'', p_2'', ...]; h_3=...```.
The p_1' and p_1'' can be anything, e.g., [1,1,1] or [1,0,0].
If we choose the p's to be discretized and binarized version of 2d images generated
by 2d Gaussian processes, we can get a discretized and binarized version of
the active-sensing task studied in [this paper](http://scottchenghsinyang.com/paper/Yang-eLife-2016.pdf).
The prior will now be over both h and p, P(h,p).
The likelihood will take the form P(y|x,p,h).
The posterior will be P(p,h|xs,ys).
Code expected information gain for this hierarchical concept space.
See equations (1)--(3) and descriptions around the equations in
[this paper](http://scottchenghsinyang.com/paper/YangShafto_CogSci_2017_final.pdf)
for some details.

## Remarks
This tutorial is made for a session in the
[online machine learning PhD summer school at the Technical University of Denmark (DTU)](http://www2.compute.dtu.dk/courses/02901/).
My introduction slides can be found in this repo.
