import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from model import GPT
from tokenizer import BPETokenizer

text = open("data/code.txt").read()
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.build_vocab(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

seq_len = 128
x, y = [], []
for i in range(len(data)-seq_len):
    x.append(data[i:i+seq_len])
    y.append(data[i+1:i+seq_len+1])
x = torch.stack(x)
y = torch.stack(y)

dataset = TensorDataset(x,y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(vocab_size=len(tokenizer.stoi), embed_dim=256, heads=8, ff_hidden=1024, layers=6, seq_len=seq_len).to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()

for epoch in range(10):
    for xb,yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)), yb.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch+1}: {loss.item()}")
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")
