import argparse, os, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# tiny dataset loader
# -------------------------
DEFAULT_CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune;
Or to take arms against a sea of troubles,
And by opposing end them.
"""

def load_text():
    if os.path.exists("input.txt"):
        with open("input.txt", "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_CORPUS.strip()

# -------------------------
# data utils (char-level)
# -------------------------
class CharData:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def get_batch(self, split, block_size, batch_size, device):
        n = int(0.9*len(self.data))
        data = self.data[:n] if split=="train" else self.data[n:]
        ix = torch.randint(low=0, high=len(data)-block_size-1, size=(batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    def decode(self, idxs):
        return "".join(self.itos[int(i)] for i in idxs)

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

# -------------------------
# tiny GPT model
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # buffer causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))                   # (B, nh, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v                                                               # (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)                             # (B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=2, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # init (xavier is fine for tiny nets)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence longer than block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1,T)

        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -------------------------
# training
# -------------------------
def train(model, data, block_size, batch_size, steps, lr, device, eval_interval=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_eval = float('inf')
    model.train()
    for step in range(1, steps+1):
        x, y = data.get_batch("train", block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == 1 or step == steps:
            with torch.no_grad():
                xval, yval = data.get_batch("val", block_size, batch_size, device)
                _, vloss = model(xval, yval)
            print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}")
            if vloss.item() < best_eval:
                best_eval = vloss.item()

# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiny SLM (mini-GPT) from scratch")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--generate-len", type=int, default=200)
    parser.add_argument("--seed", type=str, default="To be")
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(1337)

    device = "cpu"  # keeps it laptop-friendly
    text = load_text()
    data = CharData(text)

    print(f"Loaded corpus len={len(text)}, vocab_size={data.vocab_size}")
    model = TinyGPT(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    t0 = time.time()
    train(model, data, args.block_size, args.batch_size, args.train_steps, args.lr, device)
    t1 = time.time()
    print(f"Training done in {t1 - t0:.1f}s")

    # generate sample
    start = data.encode(args.seed).unsqueeze(0).to(device)
    out = model.generate(start, max_new_tokens=args.generate-len)
    txt = data.decode(out[0].tolist())
    print("\n=== SAMPLE ===")
    print(txt)
    print("==============")

if __name__ == "__main__":
    main()