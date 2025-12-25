import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 再現性確保
torch.manual_seed(42)

# ==========================================
# 1. ChemicalLLM モデル定義
# ==========================================

class ChemicalLLM(nn.Module):
    def __init__(self, vocab_size, num_basis=32, decay=0.1, alpha=0.2): # alphaを0.2に下げて安定化
        super().__init__()
        self.N = num_basis
        self.decay = decay
        self.alpha = alpha
        
        self.embedding = nn.Embedding(vocab_size, num_basis)
        # 初期値を少し小さめ(0.05)にして、暴走を防ぐ
        self.W = nn.Parameter(torch.randn(num_basis, num_basis, num_basis) * 0.05)
        self.decoder = nn.Linear(num_basis, vocab_size)

    def forward(self, input_ids, hidden=None):
        batch_size, seq_len = input_ids.shape
        if hidden is None:
            hidden = torch.zeros(batch_size, self.N, device=input_ids.device)
        
        outputs = []
        for t in range(seq_len):
            word_input = torch.relu(self.embedding(input_ids[:, t]))
            
            # Mixing
            x_mixed = (1 - self.decay) * hidden + word_input
            
            # Reaction
            interaction = torch.einsum('ijk,bi,bj->bk', self.W, x_mixed, x_mixed)
            
            # Update
            x_pre = x_mixed + self.alpha * interaction
            x_activated = torch.relu(x_pre)
            
            # Normalize
            sum_x = torch.sum(x_activated, dim=1, keepdim=True)
            hidden = x_activated / (sum_x + 1e-8)
            
            outputs.append(self.decoder(hidden).unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden

# ==========================================
# 2. データセット
# ==========================================
idx2word = {0: '.', 1: 'cat', 2: 'dog', 3: 'bird', 4: 'fish', 
            5: 'eat', 6: 'fly', 7: 'swim', 8: 'meat', 9: 'sky', 10: 'sea'}
word2idx = {v: k for k, v in idx2word.items()}
VOCAB_SIZE = len(idx2word)

sentences = [
    ["cat", "eat", "fish", "."],
    ["dog", "eat", "meat", "."],
    ["bird", "fly", "sky", "."],
    ["fish", "swim", "sea", "."],
    ["cat", "eat", "meat", "."]
]
train_data = [[word2idx[w] for w in s] for s in sentences]
train_tensor = torch.tensor(train_data, dtype=torch.long)

# ==========================================
# 3. 学習と評価
# ==========================================

def train_and_evaluate():
    model = ChemicalLLM(vocab_size=VOCAB_SIZE, num_basis=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("--- Start Training (Stable Version) ---")
    # Epoch数を少し多めに確保
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

    # --- 生成テスト ---
    print("\n--- Generation Test ---")
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
                if word == ".": break
                current_id = torch.tensor([[next_id]])
            print("")

if __name__ == "__main__":
    train_and_evaluate()