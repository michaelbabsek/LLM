import torch
from mpmath.ctx_mp_python import return_mpc

from model import Transformer, ModelArgs
from tokenizer import Tokenizer

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    device = get_device()
    tokenizer = Tokenizer()

    args = ModelArgs(vocab_size=len(tokenizer))
    model = Transformer(args=args).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params:,}")

    tokens = tokenizer.encode("Künstliche Intelligenz ist", add_bos=True)
    output = model.generate(tokens, device=device, max_token_length=10)
    print(tokenizer.decode(output, skip_special=True))