import torch
from model import GPT
from tokenizer import BPETokenizer

text = open("data/code.txt").read()
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.build_vocab(text)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(vocab_size=len(tokenizer.stoi), embed_dim=256, heads=8, ff_hidden=1024, layers=6, seq_len=128).to(device)
model.load_state_dict(torch.load("checkpoint_epoch10.pth"))
model.eval()

start_text = "def "
input_ids = torch.tensor([tokenizer.encode(start_text)], device=device)
generated = model.generate(input_ids, max_len=200, temperature=0.8)
print(tokenizer.decode(generated[0].tolist()))
