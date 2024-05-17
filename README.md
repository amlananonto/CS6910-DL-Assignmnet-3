This repository holds the materials for the third assignment of the CS6910 - Deep Learning course at IIT Madras.

I've developed an Encoder-Decoder Architecture, both with and without Attention Mechanism, to perform Transliteration on the provided Akshanrankar Dataset. These models were implemented using RNN, LSTM, and GRU cells from PyTorch.



To view the report, click [here](https://api.wandb.ai/links/ge22m012_omlan/x2d7w60o).

## Encoder

The encoder consists of a simple LSTM, RNN, or GRU cell. It takes a sequence of characters as input and produces a sequence of hidden states. The context vector for the decoder is derived from the hidden state of the last time step.

## Decoder

Similar to the encoder, the decoder comprises a simple LSTM, RNN, or GRU cell. It takes the encoder's hidden state and the output from the previous time step as input. The decoder generates a character sequence, utilizing an additional fully connected layer and log softmax to predict the next character.

## Attention Mechanism

The attention mechanism is implemented using the dot product approach. It computes attention scores by taking the weighted sum of softmax values derived from dot products between decoder and encoder hidden states. These attention values are concatenated with decoder hidden states and processed through a fully connected layer to obtain the decoder's output.

## Dataset

The dataset used is the Aksharankar Dataset provided by the course, focusing on the Tamil language. It includes `train.csv`, `valid.csv`, and `test.csv` files, each containing English and Tamil word pairs. These pairs serve as input and output strings for the transliteration task.
