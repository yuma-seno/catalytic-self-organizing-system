import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main() -> None:
    """MNIST 用のフレームワークオーバーヘッドのみを測るためのスクリプト。

    - torch / torchvision / DataLoader などの初期化
    - MNIST のダウンロードと DataLoader の構築
    - いくつかのバッチをデバイス上に転送

    モデルやオプティマイザは一切定義せず、
    Python ランタイムや PyTorch, データローダ等の
    共通オーバーヘッドのみを含む。
    """

    INPUT_DIM = 28 * 28
    BATCH_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mnist_framework_only] Using Device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # データローダの挙動をある程度再現するため、
    # 数バッチ分だけ CPU/GPU へ転送しておく。
    num_warmup_batches = 10
    print(f"[mnist_framework_only] Warming up DataLoader for {num_warmup_batches} batches ...")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if batch_idx + 1 >= num_warmup_batches:
            break

    print("[mnist_framework_only] Done. Exiting without model construction.")


if __name__ == "__main__":
    main()
