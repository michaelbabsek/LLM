This is my first Large Language Model. 
The aim of this project is to train a ~143M parameter model (with weight tying) first and, if that succeeds, a model with ~1-3B parameters.
Currently I'm using the OpenWebText dataset.

My Goal is to make this project very readable and educational with much flexibility.

## Model

GPT-style decoder-only transformer (`model.py`): 16 blocks, 768 dims, 8 heads, 1024 context, RMSNorm, GPT-2 tokenizer
plus a few special tokens (vocab padded to 50,304). The output layer shares its weights with the token embedding,
which brings the total to 143,254,272 parameters.

## Usage

```bash
# 1. prepare data: tokenizes 50% of OpenWebText into ./dataset/train.bin and val.bin (uint16)
python dataset.py
python check_data.py            # token counts, max token id, BOS/EOS share, decoded samples

# 2. smoke test on a laptop (small 29M model, 300 steps, a few minutes on Apple Silicon)
python train.py --config config.smoke.yml

# 3. full training on a 24 GB CUDA GPU (bf16 + torch.compile), see the comments in config.gpu.yml
#    for adjusting train_iters after measuring tokens/s
python train.py --config config.gpu.yml

# 4. generate text for five fixed prompts into samples.md
python sample.py --ckpt ./checkpoints/gpu/best.pt
```

Training logs to Weights & Biases (`wandb login` first, or `WANDB_MODE=offline`).
Checkpoints go to `training.ckpt_dir`: `best.pt` (lowest validation loss) and `last.pt` (every `ckpt_interval`
steps and on Ctrl+C). Running the same command again resumes from the newer of the two, including optimizer,
lr schedule, data order and the wandb run.

## Results

_TODO_
