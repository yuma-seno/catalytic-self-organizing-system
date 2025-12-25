import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


torch.manual_seed(42)


# ==========================================
# 1. RNN ベースラインモデル
# ==========================================


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None):
        emb = self.embedding(input_ids)
        out, hidden = self.rnn(emb, hidden)
        logits = self.decoder(out)
        return logits, hidden


# ==========================================
# 2. データセット
# ==========================================


idx2word = {0: ".", 1: "cat", 2: "dog", 3: "bird", 4: "fish", 5: "eat", 6: "fly", 7: "swim", 8: "meat", 9: "sky", 10: "sea"}
word2idx = {v: k for k, v in idx2word.items()}
VOCAB_SIZE = len(idx2word)

sentences = [
    ["cat", "eat", "fish", "."],
    ["dog", "eat", "meat", "."],
    ["bird", "fly", "sky", "."],
    ["fish", "swim", "sea", "."],
    ["cat", "eat", "meat", "."],
]
train_data = [[word2idx[w] for w in s] for s in sentences]
train_tensor = torch.tensor(train_data, dtype=torch.long)


# ==========================================
# 3. 学習と評価
# ==========================================


def train_and_evaluate_rnn() -> None:
    model = RNNLanguageModel(vocab_size=VOCAB_SIZE, emb_dim=32, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("--- Start Training (RNN Baseline) ---")
    for epoch in range(501):
        optimizer.zero_grad()
        inputs = train_tensor[:, :-1]
        targets = train_tensor[:, 1:]

        outputs, _ = model(inputs)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}")

    print("\n--- Generation Test (RNN Baseline) ---")
    model.eval()
    test_words = ["bird", "dog", "fish"]

    with torch.no_grad():
        for start_word in test_words:
            current_id = torch.tensor([[word2idx[start_word]]])
            hidden = None
            print(f"Start '{start_word}': ", end="")

            for _ in range(5):
                output, hidden = model(current_id, hidden)
                next_id = output[0, -1, :].argmax().item()
                word = idx2word[next_id]
                print(f"-> {word} ", end="")
                if word == ".":
                    break
                current_id = torch.tensor([[next_id]])
            print("")


if __name__ == "__main__":
    train_and_evaluate_rnn()
