This is my first Large Language Model. 
The aim of this project is to train a ~153M parameter model (with weight tying) first and, if that succeeds, a model with ~1-3B parameters.
Currently I'm using the OpenWebText dataset.

My Goal is to make this project very readable and educational with much flexibility.

## Model

GPT-style decoder-only transformer (`model.py`): 16 blocks, 768 dims, 8 heads, 1024 context, RMSNorm (pre-norm),
attention output projection, GPT-2 tokenizer plus a few special tokens (vocab padded to 50,304). The output layer
shares its weights with the token embedding, which brings the total to 152,691,456 parameters.
As in GPT-2, the projections into the residual stream (`attn.proj`, `mlp.fc2`) are initialized with
std 0.02 / sqrt(2 * n_blocks), and weight decay applies to every matrix (including the tied embedding), but not to
norms and biases.

## Usage

```bash
# 1. prepare data: tokenizes 50% of OpenWebText into ./dataset/train.bin and val.bin (uint16)
python dataset.py
python check_data.py            # token counts, max token id, BOS/EOS share, decoded samples

# 2. smoke test on a laptop (small 30M model, 300 steps, a few minutes on Apple Silicon)
python train.py --config config.smoke.yml

# 3. full training on a 24 GB CUDA GPU (bf16 + torch.compile), see the comments in config.gpu.yml
#    for adjusting train_iters after measuring tokens/s
python train.py --config config.gpu.yml

# 4. generate text for five fixed prompts into samples.md
python sample.py --ckpt ./checkpoints/gpu/best.pt
```

Training logs to Weights & Biases (`wandb login` first, or `WANDB_MODE=offline`).
Checkpoints go to `training.ckpt_dir`: `best.pt` (lowest validation loss), `last.pt` (every `ckpt_interval`
steps and on Ctrl+C) and `step_005000.pt`, `step_010000.pt`, ... (every `keep_interval` steps, never overwritten).
Running the same command again resumes from the newer of `last.pt` and `best.pt`, including optimizer,
lr schedule, data order and the wandb run.

Each log step records `train/grad_norm` (before clipping), `diag/emb_norm` (norm of the tied embedding matrix) and
`diag/max_abs_logit` (of the last batch), which usually rise before a loss spike.

If the loss diverges, go back to a numbered checkpoint and continue with a lower learning rate:

```bash
# continues at step 5000 with half the lr schedule, in a new wandb run
python train.py --config config.gpu.yml --resume ./checkpoints/gpu/step_005000.pt --lr-scale 0.5
```

The lr scale is stored in the checkpoint, later restarts keep it without passing `--lr-scale` again.
Checkpoints from before the attention output projection was added can't be loaded anymore.

## Results

_TODO_
