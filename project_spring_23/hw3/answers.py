r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 150
    hypers['seq_len'] = 64
    hypers['h_dim'] = 256
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 0.3
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.1
    start_seq = "ACT III. SCENE I. The King"
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences for training instead of using the entire text for a few important reasons.
Firstly, it enhances memory and computational efficiency. Training on the entire text simultaneously can be 
memory-intensive and computationally expensive. Splitting it into smaller sequences allows for more manageable data 
processing.
Secondly, it promotes sequential learning. Language models benefit from predicting the next character or word based on 
the preceding sequence's context. Splitting the text into sequences enables the model to capture dependencies and 
patterns within shorter contextual windows.
Furthermore, it helps address the challenge of capturing long-range dependencies. Longer dependencies are difficult to 
learn directly, but shorter sequences allow the model to focus on learning dependencies within each sequence. The 
recurrent connections in the model retain information from previous sequences, capturing broader dependencies.
Lastly, splitting into sequences enables the generation of corresponding labels. Shifting the sequence by one character 
creates labels that train the model to predict the next character in the original text, facilitating effective model 
training.
In conclusion, splitting the corpus into sequences improves efficiency, captures dependencies, and enables appropriate 
label generation for training the model.
"""

part1_q2 = r"""
The length of the output text is influenced by the hidden state size, while the hidden state itself is independent of 
the input size. This independence can lead to the generated text displaying a memory longer than the sequence length. 
This phenomenon occurs because the hidden layers, spanning multiple batches, can influence each batch of characters. 
Consequently, we have limited control over the exact length of the generated text due to the complex interplay between 
hidden states and the sequential nature of the model.
"""

part1_q3 = r"""
By maintaining the order of batches during training, we preserve the sequential information present in the corpus. 
This allows the model to capture and learn from the dependencies and patterns that arise from the natural flow of the 
text.
"""

part1_q4 = r"""
**1.** We lower the temperature for sampling in order to allow the softmax function to output more determined decisions 
and reduce the uniformity of the probability distribution. By lowering the temperature, the model's predictions become 
sharper, prioritizing the most likely outcomes based on its learned patterns. This helps the model generate text that 
aligns closely with the training data and improves coherence and conservatism in text generation. Additionally, as the 
temperature decreases over time during training, the model relies less on new data and remembers older information from 
the text.

**2.** When the temperature is set to a very high value, the distribution output by the softmax becomes closer to 
uniform. This high randomness and lack of determination in the decision-making process can negatively impact the 
learning process. The model is more likely to switch between different data points frequently, causing it to have 
difficulty in establishing meaningful connections between different parts of the text. This high temperature can hinder 
the model's ability to learn coherent patterns and generate text that aligns well with the expected language structures.

**3.** On the other hand, when the temperature is set to a very low value, the distribution output by the softmax 
becomes more spiky. This means that characters with high probabilities receive even higher scores, while others receive 
lower scores. The low temperature helps the model make more determined decisions during training, as it focuses on the 
most probable outcomes and reduces randomness. However, this can lead to a limited exploration of the training data and 
make it challenging to update the model's memory with new information. The model becomes highly reliant on the old text,
potentially limiting its ability to adapt to new patterns or generate diverse text.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 100
    hypers['z_dim'] = 40
    hypers['x_sigma2'] = 0.0005
    hypers['learn_rate'] = 0.0005
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
The $\sigma^2$ (`x_sigma2` in the code) hyperparameter is crucial for determining the variance in the latent space. 
With a larger x_sigma2 value, similar photos from the instance space are mapped to distinct points in the latent space. 
This expands the "area" of similar photos surrounding each point, resulting in more randomness in the generated images, 
allowing for a greater variety of poses and backgrounds.

In contrast, a smaller x_sigma2 restricts the mapping of similar photos to a smaller "area" in the latent space. This 
leads generated images closely resembling those in the dataset, with reduced pose and background variation.

In summary, adjusting the x_sigma2 hyperparameter balances the tradeoff between reconstruction loss and KLD loss. 
It determines how photos from the instance space are mapped to points in the latent space, influencing the similarity 
and diversity of the generated images. By modifying x_sigma2, we can control the level of variation and fidelity in the 
generated outputs.
"""

part2_q2 = r"""
**1.** The reconstruction loss, which calculates the negative log-likelihood expected for each data point, such as 
the i-th datapoint. This loss term evaluates the decoder's capacity to reconstruct the original data accurately based 
on the encoder's distribution over representations. By penalizing deviations between the model's output and the true 
data, it encourages the model to improve the quality of reconstruction by adjusting its parameters.
The KL divergence loss, which measures the difference between the encoder's distribution over representations
and the prior distribution. This loss term evaluates the encoder's capacity to learn a meaningful representation of
the data. By penalizing deviations between the encoder's distribution and the prior distribution, it encourages the 
model to learn a representation that aligns with the prior distribution.

**2.** The KL loss term regulates the latent-space distribution, encouraging it to resemble a 
predefined prior distribution. This alignment ensures a more controlled and meaningful generation process, as the latent 
variables follow a known distribution and allow for interpretable synthesis of new data.

**3.** In our case, the prior distribution is a standard normal distribution, with gaussian distribution we can sample
from. This allows us to generate new data points by sampling from the latent space, as we can easily map the latent
space to the instance space. Additionally, the KL loss term is easier to calculate when the prior distribution is a
standard normal distribution, as the KL divergence between two gaussian distributions can be calculated analytically.
"""

part2_q3 = r"""
The goal is to bring the distribution $P(X)$ as close as possible to its upper bound. However, since we do not know the 
evidence distribution used to generate new data points, we aim to estimate it by maximizing its likelihood and 
minimizing the divergence from the real distribution. In this process, the KL loss term associated with 
$q(Z|X) || p(Z)$ is often considered small in proportion to the term representing $P(X)$. Hence, it can be dropped or 
disregarded. Ultimately, the objective is to approximate the evidence distribution and maximize its closeness to the 
true distribution in order to improve the overall model performance.
"""

part2_q4 = r"""
To ensure numerical stability, we model the logarithm of the latent space variance instead of the variance itself. This 
approach spreads the variance values across a wider numerical range and avoids issues associated with small positive 
values. Additionally, the logarithm function's differentiability allows for easy calculation of derivatives during 
backpropagation.

For instance, when working with small variance values close to zero, directly modeling them can lead to numerical 
instability. However, by taking the logarithm, we transform these values into a more manageable range, enabling 
smoother computations and stable training.

In summary, modeling the logarithm of the latent space variance enhances numerical stability and facilitates gradient 
calculations. This approach is particularly valuable when dealing with small variance values, as it improves the 
overall performance and reliability of the model.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
The plots indicate that allowing all layers, rather than just the last two, to adapt during fine-tuning improves model 
performance. This approach results in a more significant drop in loss and a boost in accuracy.
Modifying other model elements, like multi-head attention, can further increase flexibility and performance, 
especially when tackling new, dissimilar tasks.
However, a fine-tuned model may not always outperform in every scenario. Its success depends on various factors such as 
the similarity of the pre-training and fine-tuning tasks, available data volume, and task complexity. In certain 
situations, like when the new task is quite different or when there's plenty of task-specific data, a model trained from 
scratch might be more effective. Similarly, for simpler tasks, a basic from-scratch model could match a fine-tuned 
model's performance.
"""

part3_q2 = r"""
Assuming there is a typo and the meaning was **unfreeze** and not *freeze*, then the answer is:
Yes, the model would still be able to succesfully fine-tune to this task. The results would not necessarily be worse.
if we unfroze and fine-tuned internal layers. Adjusting other internal components, such as multi-head attention, will 
add flexibility to the model, potentially improving its performance on new tasks, especially those vastly different from 
the original training task. However, the model would require more training data to learn the new task, as it would need 
to learn the new patterns from scratch. Additionally, the model would need to be trained for a longer period of time, 
as it would need to learn the new task from scratch. The decision to unfreeze specific layers for fine-tuning involves 
some trial and error, necessitating a solid grasp of your data, model, and an effective training process.
"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
