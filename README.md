# Linear-Polynomial-Regression
Linear and Polynomial Regression with Bias-Variance Tradeoff

The Mean Squared Error (MSE) is used as the loss function:
            1/n sumation of(y - y^)^2
            n= number of inputs
            y = actual value
            y cap(y^) = predicted value 
* Why squared error is used for Penalizes large errors and Prevents error cancellation

* What minimizing loss means
  Loss represents how far off the model's predictions are from actual targets and minimizing the loss function can be done by the  training process of reducing the error between predicted and actual values by optimizing model parameters.

*How the model parameters influence the loss
  Model parameters  directly effect a model's output, and each every parameter has it's own influence on model peformance and output therefore its prediction error or loss, acting as the primary lever for optimization.

*Why does training error always decrease with higher polynomial degree?
   Training error always decreases as the polynomial degree increases because higher-degree polynomials have more parameters and  greater model flexibility to fit the training data . This  is a direct result of the model's increased capacity to memorize  curve-fit even the noise within the training dataset.


*Why does test error behave differently?
  Test error behaves differently because it measures a model's performance on unseen data, whereas training error measures performance on data the model has already memorized. It acts as a gauge for generalization, often decreasing initially with model complexity before increasing due to over-fitting noise, creating a U-shape curve. 

*At what point does the model start overfitting, and how can you tell?
   Overfitting in a machine learning model typically starts when the model continues to remember from the training data more rather than learning patterns in training data and i can tell model start overfitting when model is performing good on training data and giving bad perform on testing data.

*If you had to choose one polynomial degree, which would it be and why?
   The optimal polynomial degree is the one that achieves the lowest testing error while keeping training and testing errors close to each other.  a moderate polynomial degree such as degree 4 gives the best balance.This degree captures the non-linear structure of the data without fitting noise, resulting in good performance and  it represents the best tradeoff between bias and variance.
  
