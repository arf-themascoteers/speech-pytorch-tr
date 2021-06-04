import torch
from encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
model = Encoder(src_vocab_size, 512, 3, 4, device, 4, dropout=0, max_length=100).to(device)
out = model(x, trg[:, :-1])
print(out.shape)