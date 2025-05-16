import sys
import time

import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer


tokenizer = Tokenizer()
args = ModelArgs(vocab_size=len(tokenizer))
model = Transformer(args=args)
model.load_state_dict(torch.load('model.pt'))

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:_}")

text = ""
sequence = tokenizer.encode(text)
print(sequence)

for i in range(30):
    sequence_tensor = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    logits = model(sequence_tensor)
    next_token_logits = logits[:, -1, :]  # (1, vocab_size)
    predicted_id = next_token_logits.argmax(dim=-1)
    sequence.append(predicted_id)
    sys.stdout.write("\r" + tokenizer.decode(sequence))
    sys.stdout.flush()
    time.sleep(0.05)