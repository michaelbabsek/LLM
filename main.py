from model import Transformer, ModelArgs
from tokenizer import Tokenizer
from train import get_device


if __name__ == "__main__":
    device = get_device()
    tokenizer = Tokenizer()
    args = ModelArgs(vocab_size=len(tokenizer))
    model = Transformer(args=args).to(device)

    input = tokenizer.encode("Hallo")
    output = model.generate(input, device=device, max_token_length=50)
    print(tokenizer.decode(output))