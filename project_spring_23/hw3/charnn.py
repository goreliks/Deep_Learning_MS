import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    char_to_idx = {}
    idx_to_char = {}
    chars = sorted(set(text))
    for i, char in enumerate(chars):
        char_to_idx[char] = i
        idx_to_char[i] = char
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = re.sub(f"[{''.join(chars_to_remove)}]", "", text)
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    result = torch.zeros(len(text), len(char_to_idx), dtype=torch.int8)
    for i, char in enumerate(text):
        result[i, char_to_idx[char]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    result = ""
    for row in embedded_text:
        result += idx_to_char[row.argmax().item()]
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    num_samples = len(text[:-1]) // seq_len
    one_hot_text = chars_to_onehot(text, char_to_idx)
    num_relevant_chars = num_samples * seq_len
    samples = one_hot_text[:num_relevant_chars, :]
    samples = samples.reshape(num_samples, seq_len, -1).to(device)
    labels = one_hot_text[1:num_relevant_chars+1, :]
    labels = ((labels == 1).nonzero(as_tuple=False))[:, 1]
    labels = labels.reshape(num_samples, seq_len).to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    y = y / temperature
    result = torch.softmax(y, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    chars_to_generate = n_chars - len(start_sequence)
    with torch.no_grad():
        h = None
        embed = chars_to_onehot(start_sequence, char_to_idx)
        embed = embed.to(dtype=torch.float, device=device)

        for idx in range(chars_to_generate):
            scores, h = model(embed.unsqueeze(dim=0), hidden_state=h)
            dist = hot_softmax(scores[0, -1, :], dim=0, temperature=T)
            next_idx = torch.multinomial(dist, 1).item()
            next_char = idx_to_char[next_idx]
            out_text = out_text + next_char
            embed = chars_to_onehot(next_char, char_to_idx)
            embed = embed.to(dtype=torch.float, device=device)
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        batch_size_data = (len(self.dataset) // self.batch_size) * self.batch_size
        idx = torch.arange(batch_size_data)
        idx = idx.reshape(self.batch_size, -1).transpose(0, 1).flatten().tolist()
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        for layer in range(self.n_layers):
            lst = []

            z_x = nn.Linear(in_dim, h_dim, bias=False)
            lst.append(z_x)
            self.add_module(f'z_x_{layer}', z_x)  # add module
            z_h = nn.Linear(h_dim, h_dim, bias=True)
            lst.append(z_h)
            self.add_module(f'z_h_{layer}', z_h)  # add module

            r_x = nn.Linear(in_dim, h_dim, bias=False)
            lst.append(r_x)
            self.add_module(f'r_x_{layer}', r_x)  # add module
            r_h = nn.Linear(h_dim, h_dim, bias=True)
            lst.append(r_h)
            self.add_module(f'r_h_{layer}', r_h)  # add module

            g_x = nn.Linear(in_dim, h_dim, bias=False)
            lst.append(g_x)
            self.add_module(f'g_x_{layer}', g_x)  # add module
            g_h = nn.Linear(h_dim, h_dim, bias=True)
            lst.append(g_h)
            self.add_module(f'g_h_{layer}', g_h)  # add module

            in_dim = self.h_dim
            self.layer_params.append(lst)

        self.add_module("dropout", nn.Dropout(dropout))  # add dropout
        self.add_module("out_layer", nn.Linear(in_dim, out_dim, bias=True))  # output layer

        for m in self.modules():
            if type(m) is not nn.Linear:
                continue
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
                # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        for idx in range(self.n_layers):
            hidden = layer_states[idx]
            next_layer_input = []

            for t in range(seq_len):
                X_t = layer_input[:, t, :].type(torch.float)
                z_x, z_h, r_x, r_h, g_x, g_h = self.layer_params[idx]

                Z_t = torch.sigmoid(z_x(X_t) + z_h(hidden))
                R_t = torch.sigmoid(r_x(X_t) + r_h(hidden))
                G_t = torch.tanh(g_x(X_t) + g_h((R_t * hidden)))
                hidden = Z_t * hidden + (1 - Z_t) * G_t
                next_layer_input.append(hidden)

            layer_states[idx] = hidden
            layer_input = self.dropout(torch.stack(next_layer_input, 1))

        hidden_state = torch.stack(layer_states, 1)
        layer_output = self.out_layer.forward(layer_input)
        # ========================
        return layer_output, hidden_state
