import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MLP_MNIST(nn.Module):
    """Simple MLP baseline for MNIST.

    Uses two hidden layers. Hidden width is chosen so that
    the total parameter count is of the same order as CSOS_MNIST,
    but the architecture is conventional.
    """

    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_mnist_mlp() -> None:
    print("\n=== Baseline: MLP on MNIST ===")

    INPUT_DIM = 28 * 28
    EPOCHS = 10
    BATCH_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP_MNIST(input_dim=INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
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

        acc = 100.0 * correct / total
        print(f"Epoch {epoch}: Acc {acc:.2f}% | Loss {epoch_loss/len(train_loader):.4f}")

    # Test
    print("\n--- Baseline MLP: Final Test ---")
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

    print(f"Test Accuracy (MLP): {100.0 * correct / total:.2f}%")


if __name__ == "__main__":
    train_mnist_mlp()
