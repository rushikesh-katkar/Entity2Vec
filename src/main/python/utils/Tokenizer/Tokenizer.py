import torch

class simpleTokenizer:
    def __init__(self, documents, device=None):

        vocab = set()
        self.itos = {}
        self.stoi = {}

        token_num = 0

        for tracks in documents.values():
            for ele in tracks:
                if ele not in vocab:
                    vocab.add(ele)
                    self.itos[token_num] = ele
                    self.stoi[ele] = token_num
                    token_num += 1

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

    def encode(self, x):
        return torch.tensor(
            [self.stoi[s] for s in x],
            device=self.device
        ).unsqueeze(0)

    def decode(self, x):
        x = x.detach().cpu().reshape(-1)
        return [self.itos[i.item()] for i in x]