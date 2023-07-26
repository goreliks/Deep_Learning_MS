r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

**1.False.**
The test set is used to estimate the out-of-sample error, not the in-sample error. In-sample error is evaluated on the 
training set, which is the set of data used to fit or train the model. The test set is a separate set of data not used 
during training, and it gives an estimate of how well the model will perform on new, unseen data, 
which is the out-of-sample error.

**2.False.**
While it is technically possible to split the data into two disjoint subsets in many ways, not all splits would 
be equally useful or appropriate. A good train-test split should reflect the same distribution and have similar
characteristics. Random splits are typically used to ensure this. Also, it's important to ensure that
the test set is large enough to provide a reliable estimate of the model's performance on new data,
but not so large that it significantly reduces the amount of training data available.

**3.True.**
The test set should not be used during cross-validation. The goal of cross-validation is to estimate the generalization 
error of the model, and this is done by partitioning the training data into smaller subsets or "folds". 
One of these folds is used as a validation set to evaluate the model, while the rest are used to train the model. 
This process is repeated with each fold serving as the validation set once. The test set is kept separate and 
is only used after the model has been finalized, to estimate its performance on new, unseen data.

**4.True.**
After performing cross-validation, we indeed use the validation-set performance of each fold as a proxy for the 
model’s generalization error. The idea behind cross-validation is to simulate the process of training a model and 
then testing it on unseen data. By averaging the performance of the model on each validation set, we can estimate 
how well the model is likely to perform on new data. This gives us a measure of the model's generalization error, 
which is how well we can expect the model to perform on data it hasn't seen before.

"""

part1_q2 = r"""
**Your answer:**

While my friend's idea to add a regularization term is a good strategy to combat overfitting, 
the approach of using the test set to tune the hyperparameter $\lambda$ is not justified.
The test set should ideally be used only once, after all model decisions (including hyperparameter tuning) 
have been made, to give an unbiased estimate of the model's performance on unseen data. 
By using the test set to tune $\lambda$, my friend is essentially using the test set as part of the model 
building process, which could lead to overfitting to the test set. This means that the performance of the model 
on the test set may be overly optimistic, and not truly reflective of how the model would perform on new, unseen data.

Instead, my friend should consider splitting the original training set further into a smaller training set and 
a validation set. The model can then be trained on the smaller training set with different values of $\lambda$, 
and the performance can be evaluated on the validation set. The value of $\lambda$ that results in the best performance 
on the validation set should be chosen. Once this is done, the model can be evaluated on the test set to get a final, 
unbiased estimate of the model's performance.

Alternatively, cross-validation could be used instead of a single validation set. In cross-validation, the training 
data is split into multiple folds, and the model is trained and evaluated multiple times, with each fold serving as 
the validation set once. The value of $\lambda$ that results in the best average performance across all folds 
can be chosen. This approach has the benefit of making full use of the available training data, 
while still providing a robust way to tune hyperparameters.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
Increasing the value of k in KNN does not always lead to improved generalization for unseen data. 
The choice of the optimal k value depends on the dataset and its underlying characteristics.

Based on the results we got, it appears that increasing the value of k beyond k=5 (k=5 mean accuracy is lower 
because of result of one fold, but in general it performed even better than k=3 on some folds) does not 
necessarily lead to improved generalization for unseen data. In fact, the average accuracies tend to decrease as 
k increases from 5 to 50.

When k is too small (e.g., k = 1), the model tends to be highly influenced by the noise or outliers 
in the training data. This can result in overfitting, where the model becomes too specific to the training set and 
performs poorly on unseen data. In such cases, the model may have difficulty capturing the underlying patterns and 
may exhibit high variance.

On the other hand, when k is too large, for example 2500 samples (the whole validation set) the model 
becomes overly generalized. It may start considering too many irrelevant or distant points, leading to underfitting. 
The model's decision boundaries become smoother, and it may fail to capture local patterns or intricacies in the data. 
This can result in high bias and limited ability to discriminate between different classes or clusters.

"""

part2_q2 = r"""
**Your answer:**

**1. Training on the entire train-set with various models and selecting the best model with respect to train-set 
accuracy:**
Using k-fold cross-validation (CV) is better than training on the entire train-set and selecting the best model 
based on train-set accuracy alone. When selecting the best model based on train-set accuracy, there is a risk of 
overfitting, where the model becomes overly specialized to the training data and may not generalize well to unseen data. 
Train-set accuracy can be an overly optimistic estimate of the model's performance since it evaluates the model on the 
same data it was trained on. Therefore, relying solely on train-set accuracy for model selection may result in choosing 
a model that performs well on the training data but poorly on unseen data.
K-fold CV addresses this issue by providing a more reliable estimate of a model's performance on unseen data. 
It involves dividing the train-set into k subsets or folds and iteratively training and evaluating the model k times. 
Each time, one of the folds is held out as a validation set, while the remaining k-1 folds are used for training. 
This process ensures that the model is evaluated on data that it has not seen during training, providing a more 
realistic estimate of its performance.

By considering the average performance across all k iterations, k-fold CV provides a more robust evaluation of 
the model's generalization ability. It helps to mitigate the influence of randomness in the data splitting process 
and provides a more representative estimate of the model's performance on unseen data. Therefore, using k-fold 
CV for model selection is preferred over relying solely on train-set accuracy.

**2. Training on the entire train-set with various models and selecting the best model with respect to test-set 
accuracy:**
Using k-fold CV is also preferable to training on the entire train-set and selecting the best model based on 
test-set accuracy alone. 
While test-set accuracy provides an estimate of the model's performance on unseen data, it is still influenced 
by the specific composition of the test set. Random variations in the test set composition can lead to variability 
in the test-set accuracy, making it an unreliable criterion for model selection.
K-fold CV addresses this limitation by evaluating the model on multiple validation sets, each representative of 
different subsets of the data. By averaging the performance across these validation sets, k-fold CV provides a more 
stable estimate of the model's generalization ability. It helps to reduce the impact of random variations in the test 
set and provides a more reliable basis for model selection.

Furthermore, by using k-fold CV, the entire train-set is utilized for both training and evaluation, 
maximizing the use of available data. This can lead to a more comprehensive assessment of model performance and 
better selection of the most suitable model.

Overall, k-fold CV offers a more robust and reliable approach to model selection compared to relying solely on 
train-set or test-set accuracy. It provides a more realistic estimate of the model's generalization ability and 
helps to mitigate the impact of data variability and overfitting.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

The selection of delta is indeed arbitrary because the model's weights can adapt to any fixed choice of delta. 
If we increase delta, the weights will just scale to compensate for this change. 
Therefore, the SVM is scale-invariant to the choice of delta.

The regularization term $$λ/2 * ||W||^2$$ is used to prevent the weights from simply scaling up, 
which could lead to overfitting. The strength of regularization λ is a hyperparameter that controls this trade-off.

In summary, while delta does influence the learning dynamics, the final model should not be sensitive 
to the particular choice of delta due to the scale invariance property of SVMs.
"""

part3_q2 = r"""
**Your answer:**

**1.** When we visualized the weights of a linear model, the weights show some pattern or structure. 
These can be seen as template matching, where the weights form a template for each class, and the model checks how 
well the input aligns with these templates. The templates capture the essential features of the class they represent.
Classification errors can occur when the learned templates are not distinctive enough to distinguish between certain 
classes.
For example, we can see that the numbers '5' and '6', that we miss classified, share similar patterns causing confusion.
It might also struggle with variations within the class, like different handwriting styles, rotations, etc.

**2.** The interpretation of what a linear model is learning is quite different from a K-Nearest Neighbors (KNN) model. 
A linear model learns a global pattern in the form of weight coefficients, which apply across all input data. 
It forms a decision boundary based on these learned patterns.
On the other hand, KNN doesn't "learn" in the traditional sense. It simply memorizes the training data. 
During prediction, it finds the 'K' training examples closest to the given test point and assigns the most common 
class among these 'K' examples. KNN makes local, instance-based decisions rather than learning a global decision 
boundary like a linear model.
So, while a linear model tries to find common patterns or templates in the data, KNN relies on the similarity of 
individual instances. Both models can work well depending on the complexity and distribution of the data.

"""

part3_q3 = r"""

**1.** Based on the graph of the training set loss, I would you say that the learning rate that I chose is seems 
to be good. The loss on the training set decreases steadily, indicating that the learning rate is not too high 
(which might cause the loss to fluctuate wildly or even increase due to overshooting the optimal point during 
gradient descent), nor too low (which might cause the loss to decrease very slowly or not at all due to making 
only tiny steps during gradient descent).

In case of a too high learning rate, the loss graph might show erratic behavior - with loss values fluctuating 
up and down and no clear trend of decreasing. This is because the gradient descent steps are so large that they 
overshoot the minimum, causing the loss to increase in the next epoch.

In case of a too low learning rate, the loss might decrease very slowly or might not decrease significantly at all 
over epochs. This is because the steps of the gradient descent are so small that it either takes a very long time 
to reach the minimum, or it gets stuck in a sub-optimal point.

**2.** Based on the training and validation accuracy, the model seems to be slightly overfitted to the training set. 
The training accuracy is higher than the validation accuracy, indicating that the model is performing better on the 
training data than on unseen data. However, the difference between training and validation accuracy is not that high 
(around 3% on the last epoch), suggesting that the overfitting is not severe.

In case of underfitting, the model would perform poorly on both the training and validation datasets, indicating 
that it has not learned the underlying patterns of the data. We would expect to see both the training and validation 
losses relatively high and the accuracies relatively low.

In case of high overfitting, we would expect the training accuracy to be very high (close to 100%) and the validation 
accuracy to be significantly lower, showing that the model is almost perfect on the training data but performs poorly 
on unseen data. In such case, the validation loss would also be much higher than the training loss.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
In an ideal case, if the model fits the data perfectly, the residual plot would show a random scatter of points 
clustered around the horizontal axis.
We can see that it's not ideal, but still pretty good. The residuals are almost equally distributed above and below the 
horizontal line at zero. This indicates that the model's predictions are equally likely to be too high as too low.
The residuals appear to be randomly distributed with no obvious patterns, curves, or trends. Since if there were the
trend in the residuals, it suggests that the model isn't capturing some aspect of the data.
We still have few outliers and the spread of the residuals doesn't remain approximately constant across all levels 
of the predicted outcome. 
In summary, based on the residuals plot we got, the fitness of model looks pretty well.

When comparing the plot we got to the plot of 5 top features, we can see that residuals are a little bit skewed towards 
positive residuals, that indicates that the model's predictions are more likely to be high. Also, there is a pattern 
that we can see there (sad parabola), that suggests that the model isn't capturing some aspects of the data.
"""

part4_q2 = r"""
**1.** Yes, it is still considered a linear regression model. This might sound counter-intuitive, but the term 
"linear" in linear regression refers to linearity in the parameters, not the features. This means that the model 
is linear in the sense that the output is a linear combination of the parameters. When we add non-linear features, 
we are still combining them in a linear way with respect to the parameters. So, the model remains a linear regression 
model even though it can capture non-linear relationships in the feature space.

**2.** By adding non-linear transformations of the features to the data, we can fit a broader range of functions 
to the data. However, this does not imply that we can fit any non-linear function with this approach. 
The extent to which we can capture non-linear relationships depends on the specific transformations we apply to 
the features. For example, polynomial transformations can capture certain types of non-linearities but may not be 
appropriate for others.

**3.** Adding non-linear features to a linear classification model can make the decision boundary in the original 
feature space non-linear. This is because when we map the input data to a higher-dimensional space (where the added 
dimensions correspond to non-linear transformations of the features), the linear decision boundary in this transformed 
space corresponds to a non-linear boundary in the original input space. The decision boundary is still a hyperplane 
in the transformed feature space, because it is defined as a linear combination of the features 
(now including the non-linear transformations).
"""

part4_q3 = r"""
**1.** The regularization parameter lambda influences the strength of the regularization term. A logarithmic scale is 
useful because it covers several orders of magnitude and thus allows us to search over a wider range of possible values.
This can be especially important for hyperparameters like lambda which can exhibit quite different behaviors over 
its range. It's a way to ensure that the CV process explores both small and large values of lambda.

**2.** The model is fitted as part of a grid search cross-validation process, where each combination of hyperparameters 
is evaluated using K-fold cross-validation. In this case, the hyperparameters are the degree of the polynomial features 
and the lambda regularization parameter. There are 3 possible degrees and 20 possible lambdas, making for 60 
combinations of hyperparameters. We used value of K=3 for the 3-fold cross-validation, 
that means the model is fitted 3 * 60 = 180 times.
"""

# ==============
