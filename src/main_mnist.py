import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ==========================================
# 1. Pure CSOS Model (Memory Efficient)
# ==========================================
class CSOS_MNIST(nn.Module):
    def __init__(self, input_dim, num_basis=64, steps=3, decay=0.1, alpha=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.steps = steps
        self.decay = decay
        self.alpha = alpha

        # Encoder: 画像(784) -> 反応場(64)
        # 次元を一気に圧縮するが、反応場のダイナミクスで情報を復元・維持する
        self.encoder = nn.Linear(input_dim, num_basis, bias=True)
        
        # Reaction Tensor: 64x64x64 の相互作用
        # 初期値を適切に設定し、反応が死なないようにする
        self.W = nn.Parameter(torch.randn(num_basis, num_basis, num_basis) * 0.05 + 0.01)
        
        # Readout: 反応場 -> 10クラス
        self.readout = nn.Linear(num_basis, 10)

    def forward(self, x):
        # 1. Injection
        encoded = self.encoder(x)
        x_curr = F.leaky_relu(encoded, negative_slope=0.01)
        
        # 正規化 (濃度は非負・総和制限)
        x_curr = torch.abs(x_curr)
        x_curr = x_curr / (x_curr.sum(dim=1, keepdim=True) + 1e-8)

        # 2. Reaction Loop (Recurrent Processing)
        # 少ないニューロンでも、時間をかけて(Steps)熟成させることで表現力を高める
        for _ in range(self.steps):
            # Mixing & Decay
            x_mixed = (1 - self.decay) * x_curr
            
            # Interaction: 全結合の化学反応
            interaction = torch.einsum('ijk,bi,bj->bk', self.W, x_mixed, x_mixed)
            
            # Update
            x_pre = x_mixed + self.alpha * interaction
            x_curr = F.leaky_relu(x_pre, negative_slope=0.01)
            
            # Normalize
            x_curr = torch.abs(x_curr)
            x_curr = x_curr / (x_curr.sum(dim=1, keepdim=True) + 1e-8)

        # 3. Readout
        return self.readout(x_curr)

# ==========================================
# 2. Training Loop
# ==========================================
def train_mnist_pure():
    print("\n=== Experiment 1: Pure CSOS on MNIST (Memory Efficient) ===")
    
    # 設定
    INPUT_DIM = 28 * 28
    HIDDEN_DIM = 64 # 一般的なMLP(512~1024)に比べて圧倒的に小さい
    EPOCHS = 5
    BATCH_SIZE = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device} | Basis Size: {HIDDEN_DIM}")

    # データ準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデル
    model = CSOS_MNIST(input_dim=INPUT_DIM, num_basis=HIDDEN_DIM, steps=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # 学習ループ
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, INPUT_DIM)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch {epoch}: Acc {acc:.2f}% | Loss {epoch_loss/len(train_loader):.4f}")

    # テスト
    print("\n--- Final Test ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, INPUT_DIM)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(f"Note: Achieved with only {HIDDEN_DIM} internal states.")

if __name__ == "__main__":
    train_mnist_pure()