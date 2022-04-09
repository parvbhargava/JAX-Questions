# SRIP-Parv-Bhargava

## Question 1 
Animate bivariate normal distribution. [10 Marks]

![image](https://user-images.githubusercontent.com/58410910/162581501-880242cd-d074-4e49-bc46-dc98ebaf94cd.png)

   ●	Reproduce the above figure showing samples from bivariate normal with marginal PDFs from scratch using JAX and matplotlib.
   ●	Add interactivity to the figure by adding sliders with ipywidgets. You should be able to vary the parameters of bivariate normal distribution (mean and                 covariance matrix) using ipywidgets.
## Question 2 
Implement from scratch a sampling method to draw samples from a multivariate Normal (MVN) distribution in JAX. [10 Marks]

    ●	Your code should work for any number of dimensions but please set the number of dimensions (random variables of MVN) to 10 for this task.
    ●	You are only allowed to use jax.random.uniform. You are especially not allowed to use jax.random.normal.
    ●	You should randomly create the mean and covariance matrix to fully specify an MVN distribution.
    ●	Implement a sampling method from scratch using which you can draw samples from the specified MVN distribution.
    ●	Use your sampling method to draw multiple samples from the MVN distribution and reconstruct the parameters of your MVN distribution (mean and covariance matrix)       to confirm that your sampling method is working correctly.
## Question 3 
Implement two hidden layers neural network classifier from scratch in JAX [20 Marks]

    ●	Two hidden layers here means (input - hidden1 - hidden2 - output).
    ●	You must not use flax, optax, or any other library for this task.
    ●	Use MNIST dataset with 80:20 train:test split.
    ●	Manually optimize the number of neurons in hidden layers.
    ●	Use gradient descent from scratch to optimize your network. You should use the Pytree concept of JAX to do this elegantly.
    ●	Plot loss v/s iterations curve with matplotlib.
    ●	Evaluate the model on test data with various classification metrics and briefly discuss their implications.
## Question 4 
Bayesian Linear Regression from scratch with BlackJAX [20 Marks]

    ●	Implement Bayesian Linear Regression from scratch with any appropriate sampling method in BlackJAX.
    ●	Create your own 1d linear dataset with added noise.
    ●	Plot the learned predictive mean and 2 standard deviations around the mean like the below plot.
![image](https://user-images.githubusercontent.com/58410910/162581527-6a9c6f2c-f601-4603-8db6-3b741ce7fd00.png)
