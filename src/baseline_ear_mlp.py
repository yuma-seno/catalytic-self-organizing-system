import os

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gtts import gTTS


# ==========================================
# 1. ベースラインモデル: MLP (Logits出力)
# ==========================================


class MLP_Audio_Segmented(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


# ==========================================
# 2. データセット (3分割特徴量)
# ==========================================


class SegmentedAudioDataset(Dataset):
    def __init__(self, n_mels: int = 64, num_samples_per_class: int = 50) -> None:
        self.n_mels = n_mels
        self.output_dim = n_mels * 3
        self.data = []

        self.commands = {
            "System Start": 1.0,
            "System Stop": 0.0,
            "Music Start": 0.0,
            "Hello Gemini": 0.0,
            "Good Morning": 0.0,
        }

        os.makedirs("generated_audio", exist_ok=True)
        print("--- Generating & Processing Audio (Segmented, Baseline MLP) ---")

        for text, label in self.commands.items():
            filename = f"generated_audio/{text.replace(' ', '_')}.mp3"
            if not os.path.exists(filename):
                tts = gTTS(text=text, lang="en")
                tts.save(filename)

            y, sr = librosa.load(filename, sr=16000)

            for _ in range(num_samples_per_class):
                noise_amp = 0.005 * np.random.rand() * np.max(y)
                y_noisy = y + noise_amp * np.random.randn(len(y))

                if np.random.rand() > 0.5:
                    steps = np.random.uniform(-1.5, 1.5)
                    y_noisy = librosa.effects.pitch_shift(y=y_noisy, sr=sr, n_steps=steps)

                feature = self.extract_segmented_feature(y_noisy, sr)
                self.data.append((feature, label, text))

        print(f"Dataset created (baseline): {len(self.data)} samples. Input Dim: {self.output_dim}")

    def extract_segmented_feature(self, y, sr):
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        time_steps = melspec_db.shape[1]
        segment_len = time_steps // 3

        if segment_len == 0:
            parts = [np.mean(melspec_db, axis=1)] * 3
        else:
            part1 = np.mean(melspec_db[:, :segment_len], axis=1)
            part2 = np.mean(melspec_db[:, segment_len : 2 * segment_len], axis=1)
            part3 = np.mean(melspec_db[:, 2 * segment_len :], axis=1)
            parts = [part1, part2, part3]

        feature_vector = np.concatenate(parts)

        min_val, max_val = -80.0, 0.0
        feature_vector = (feature_vector - min_val) / (max_val - min_val)

        return feature_vector.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label, _ = self.data[idx]
        return torch.from_numpy(feature), torch.tensor([label], dtype=torch.float32)


# ==========================================
# 3. 学習実行 (重み付けLoss)
# ==========================================


def train_segmented_audio_mlp() -> None:
    N_MELS = 64
    INPUT_DIM = N_MELS * 3
    BATCH_SIZE = 16
    EPOCHS = 60
    LR = 0.003
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE} (Baseline MLP)")

    train_dataset = SegmentedAudioDataset(n_mels=N_MELS, num_samples_per_class=50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MLP_Audio_Segmented(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    pos_weight = torch.tensor([4.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("--- Training Start (Baseline MLP) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).float()
            correct += (predicted == target).sum().item()
            total_samples += target.size(0)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            acc = 100.0 * correct / total_samples
            print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    print("\n--- Final Validation (Baseline MLP) ---")
    model.eval()

    test_words = ["System Start", "Music Start", "System Stop", "Hello Gemini"]
    for word in test_words:
        filename = f"generated_audio/{word.replace(' ', '_')}.mp3"
        y, sr = librosa.load(filename, sr=16000)
        feature = train_dataset.extract_segmented_feature(y, sr)
        tensor_in = torch.from_numpy(feature).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = model(tensor_in)
            prob = torch.sigmoid(logit).item()

        print(f"Input: '{word}' -> Probability: {prob:.4f}")
        if word == "System Start":
            status = "[OK]" if prob > 0.8 else "[FAIL]"
        else:
            status = "[OK]" if prob < 0.2 else "[FAIL]"
        print(f"   => Result: {status}")


if __name__ == "__main__":
    train_segmented_audio_mlp()
