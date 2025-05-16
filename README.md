This is my first LLM, a decoder-only transformer with about 180 million parameters (comparison: GPT-2 has about 124M). It uses learned positional embeddings and makes use of the GPT-4 Tokenizer.
This repository contains code for the model, tokenizer implementation and pre-training on the wikipedia-de dataset. 

The whole thing was programmed in less than 24 hours and can be widely improved with for example rotary positional embeddings, improvements on transforming the input (special tokens, roles) and perhaps many improvements on normalization and so on.
I only trained it for maybe an hour and the loss was gradually sinking but since GPUs are expensive i left it at that. Curious on how good it could actually perform.
