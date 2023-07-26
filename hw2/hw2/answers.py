r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**1.** Our starting point is an input tensor with a shape of (128, 1024), representing 128 samples, each with 1024 features. 
This is processed through a layer, resulting in an output tensor shaped (128, 2048) - 
the same 128 samples now mapped to 2048 features.

To study the input-output relationship, we use the Jacobian matrix, which evaluates how each input impacts the output, 
quantified as partial derivatives. Carrying out this analysis across all samples gives us a 4-dimensional tensor known 
as the Jacobian.

In terms of dimensions, the Jacobian tensor is characterized as follows:

* First and third dimensions are both the batch size (128).
* The second dimension pertains to the output features (2048).
* The fourth dimension relates to the input features (1024).

This results in the Jacobian shape being (128, 2048, 128, 1024).
<br>

**2.** The size of a single-precision floating point number is 32 bits or 4 bytes.
The Jacobian tensor has a shape of (128, 2048, 128, 1024), which means it has 128 * 2048 * 128 * 1024 = 34,359,738,368 
elements.
Multiplying this by 4 bytes per element gives us a total of 137,438,953,472 bytes.
This is equivalent to 137,438,953,472 / 1024 / 1024 / 1024 = 128 GB.


.
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg = 8, 0.02, 0.005, 0.0002, 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd, lr = 1, 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**1.** Our experiments yielded expected results. A zero Dropout setting led to higher training accuracy than 
a 0.4 Dropout. This occurred because every neuron participated in learning, tuning the model to the data, which, 
however, risked overfitting. Despite this model having higher training accuracy, the model with dropout showed 
superior test accuracy.

**2.** Comparing models with low and high dropout, the low dropout model had better test accuracy. This suggests 
that high dropout values may obstruct effective weight training, slow learning speed, potentially cause underfitting, 
but appropriately chosen dropout values can help prevent overfitting.
"""

part2_q2 = r"""
Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases. 
This might occur due to the probabilistic nature of the model's output. If the model becomes more confident about 
its incorrect predictions, the cross-entropy loss (which penalizes false confidence) will increase. Simultaneously, 
the model could improve on making correct predictions more accurate, leading to increased accuracy. 
Thus, a more confident model can see an increase in both test loss and accuracy.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**1.**$Residual_block=64*(3*3*256+1)+64*(3*3*64+1)=184448$ params, 
$Bottleneck_layer=64*(1*1*256+1)+64*(3*3*64+1)+256(1*1*64+1)=70,016$ params

**2.** For a single residual block without Bottle Neck:
$Number_of_filters=256$
$Number_of_pixels=28*28$
$Number_of_calculations_per_pixel=256*3*3$
$Total_number_of_calculations=Number_of_filters*Number_of_pixels*Number_of_calculations_per_pixel=462,422,016$

For a single residual block with Bottle Neck:
$(1*1*256)*(28*28*64)+(3*3*64)*(28*28*64)+(1*1*64)*(28*28*256)=54,591,488$

**3.** The regular block (left block) uses a 3x3x256 kernel, while the bottleneck block (right block) utilizes 
a 1x1x256 kernel. "Within feature maps" refers to spatial operations, and "across feature maps" relates to operations 
across channels. Consequently, the regular block is more focused on spatial interactions within its feature maps, while 
both blocks can combine the 256-dimensional depth across feature maps equally. The larger kernel in the left block 
allows for more input data to be incorporated into each output result. However, the interpretation of the "difference 
in ability to combine the input" depends on the specific context.
"""

part3_q2 = r"""
**1.** Deeper networks offer improved efficiency in computation and parameter count as each layer learns more abstract 
representations. However, exceeding a certain depth threshold can lead to decreased test accuracy, deviating from our 
expectations. Although we anticipated better results from deeper CNNs in training accuracy and loss, the actual findings 
differed. L4 and L8, both depth tests, exhibited similar and superior outcomes compared to the other models. 
Despite not aligning with our initial belief that L16 would outperform them, L4 and L8 maintained higher generalization 
and accurate predictions. We speculate that L4 and L8 effectively extracted essential features while maintaining a 
reasonable level of generalization, resulting in accurate predictions and low loss.

**2.** When attempting to train the L16 network (the same happened in experiment 4 using L8), we encountered an 
error with dimensions of 0,0. This occurred due to excessive pooling layers, causing some dimensions to shrink to zero. 
We explored two potential solutions: ensuring the initial parameters were not too small and implementing a threshold 
for network parameters to prevent dimension shrinkage. Despite our efforts, the L16 network remained untrainable and 
produced unsatisfactory results without improvement. The issue likely stemmed from excessive regularization, causing 
weights to approach zero and hinder learning. To address this, we suggest reducing the regularization coefficient.
Deeper models amplify even minor changes, leading to a phenomenon called "Covariate shift" where input distributions 
differ from the training distribution without changes in the underlying mapping. To mitigate this, we propose 
utilizing Batch Normalization, which scales outputs, reduces internal covariate shift, and enhances learning speed.
"""

part3_q3 = r"""
The test accuracy graphs highlight the superior results achieved with K values of 32, 64, and 128. Specifically, in 
the L2 depth, K64 stands out as the best performer. For L4, there is comparable performance among features of 32, 64, 
and 128. In contrast, L8 shows a significant advantage for K128 over K64. These findings suggest that deeper layers 
enhance feature learning, leading to improved test accuracy.
In experiment 1.1, the graphs compare different depths while keeping the number of features constant. In experiment 1.2,
the focus is on varying the number of features while maintaining a constant depth. Both experiments explore depths of 
L2, L4, and L8, with features of K32 and K64. Experiment 1.2 introduces additional feature values of K128 and K256, 
excluding L16 depth.
These experiments provide valuable insights into how depth and the number of features impact the model's performance 
and test accuracy.
While on L2 all K values perform similarly, on L4, the value of K256 performs the worst, and on L8, the values of K64 
and K128 perform the best. This suggests that the number of features is more important than the depth of the network. 

"""

part3_q4 = r"""
Adding consecutive layers with the same number of features leads to decreased test accuracies, indicating a potential 
issue of overfitting. L4 performs the worst, followed by L3, L1, and L2, but the last three are relatively close.
"""

part3_q5 = r"""
Analyzing the graphs for fixed K32 with depths of L8, L16, and L32 revealed interesting insights. L16 achieved the 
highest accuracy and lowest loss among all depths, in contrast to the non-trainable L16 in experiment 1.1. However, 
L8 with K32 showed significant improvement in experiment 1.1. The inclusion of skip connections resolved challenges in 
experiment 1.1, likely due to the merging of shortcut paths with the main path through a ReLU activation.
These findings demonstrate the effectiveness of skip connections in enhancing training and addressing specific 
experiment difficulties.
Analyzing the graphs for fixed K64 with depths of L2, L4, and L8 we can see that L8 achieved the highest accuracy.
"""

part3_q6 = r"""
**1.** In order to tackle the issue of vanishing gradients in our deep network, we implemented skip connections using 
Residual Networks. Within the residual blocks, we incorporated Batch Normalization and activation layers after each 
convolution. Notably, dropouts were not employed in the main path of each block (dropouts=0). However, towards the end 
of the network, we introduced MaxPool2d followed by a dropout layer. This combination was specifically designed to 
improve the trainability of deeper layers and address challenges encountered in previous experiments.

**2.** We are out of time (and more free GPU time) so we can't continue to experiment with different hyperparameters. 
However, we our networks results are the same that achieved in expirement 1. We believe that if we had more time 
(and more GPU time) we could have achieved better results.
"""
# ==============
