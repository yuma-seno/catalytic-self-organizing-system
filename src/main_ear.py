import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import librosa
from gtts import gTTS

# ==========================================
# 1. モデル定義: CSOS (Logits出力版)
# ==========================================
class CSOS_Audio_Segmented(nn.Module):
    def __init__(self, input_dim, num_basis=128, steps=3, decay=0.1, alpha=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        
        self.encoder = nn.Linear(input_dim, num_basis, bias=True)
        # 初期値を少し強めに
        self.W = nn.Parameter(torch.randn(num_basis, num_basis, num_basis) * 0.05 + 0.02)
        self.readout = nn.Linear(num_basis, 1)

    def forward(self, x):
        # Phase 1: Injection
        encoded = self.encoder(x)
        x_current = F.leaky_relu(encoded, negative_slope=0.01)
        x_current = torch.abs(x_current)
        sum_x = torch.sum(x_current, dim=1, keepdim=True)
        x_current = x_current / (sum_x + 1e-8)

        # Phase 2: Reaction
        for _ in range(self.steps):
            x_mixed = (1 - self.decay) * x_current
            interaction = torch.einsum('ijk,bi,bj->bk', self.W, x_mixed, x_mixed)
            x_pre = x_mixed + self.alpha * interaction
            x_activated = F.leaky_relu(x_pre, negative_slope=0.01)
            x_activated = torch.abs(x_activated)
            sum_x = torch.sum(x_activated, dim=1, keepdim=True)
            x_current = x_activated / (sum_x + 1e-8)

        # Phase 3: Readout (Sigmoidはかけずに返す)
        return self.readout(x_current)

# ==========================================
# 2. データセット (3分割特徴量)
# ==========================================
class SegmentedAudioDataset(Dataset):
    def __init__(self, n_mels=64, num_samples_per_class=50):
        self.n_mels = n_mels
        # 入力次元 = メル周波数(64) × 3分割 = 192次元
        self.output_dim = n_mels * 3 
        self.data = []
        
        # コマンド定義
        self.commands = {
            "System Start": 1.0, 
            "System Stop": 0.0,
            "Music Start": 0.0,
            "Hello Gemini": 0.0,
            "Good Morning": 0.0,
        }
        
        os.makedirs("generated_audio", exist_ok=True)
        print("--- Generating & Processing Audio (Segmented) ---")
        
        for text, label in self.commands.items():
            filename = f"generated_audio/{text.replace(' ', '_')}.mp3"
            if not os.path.exists(filename):
                tts = gTTS(text=text, lang='en')
                tts.save(filename)
            
            y, sr = librosa.load(filename, sr=16000)
            
            # データ拡張ループ
            for _ in range(num_samples_per_class):
                # ノイズ注入
                noise_amp = 0.005 * np.random.rand() * np.max(y)
                y_noisy = y + noise_amp * np.random.randn(len(y))
                
                # ピッチシフト (バリエーション作成)
                if np.random.rand() > 0.5:
                    steps = np.random.uniform(-1.5, 1.5)
                    y_noisy = librosa.effects.pitch_shift(y=y_noisy, sr=sr, n_steps=steps)

                feature = self.extract_segmented_feature(y_noisy, sr)
                self.data.append((feature, label, text))
                
        print(f"Dataset created: {len(self.data)} samples. Input Dim: {self.output_dim}")

    def extract_segmented_feature(self, y, sr):
        # 1. メルスペクトログラム変換
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        
        # 2. 時間軸の長さを取得
        time_steps = melspec_db.shape[1]
        
        # 3. 3つのセグメントに分割して平均を取る
        # これにより "System"(前半) と "Start"(後半) を別の特徴として扱える
        segment_len = time_steps // 3
        
        # パディング (割り切れない場合のエラー防止)
        if segment_len == 0: 
            parts = [np.mean(melspec_db, axis=1)] * 3
        else:
            part1 = np.mean(melspec_db[:, :segment_len], axis=1)
            part2 = np.mean(melspec_db[:, segment_len:2*segment_len], axis=1)
            part3 = np.mean(melspec_db[:, 2*segment_len:], axis=1)
            parts = [part1, part2, part3]
        
        # 4. 結合 (Concatenate) -> 64 * 3 = 192次元
        feature_vector = np.concatenate(parts)
        
        # 正規化
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
def train_segmented_audio():
    N_MELS = 64
    INPUT_DIM = N_MELS * 3 # 192
    BATCH_SIZE = 16
    EPOCHS = 60
    LR = 0.003
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # データセット
    train_dataset = SegmentedAudioDataset(n_mels=N_MELS, num_samples_per_class=50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # モデル
    model = CSOS_Audio_Segmented(input_dim=INPUT_DIM, num_basis=128, steps=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 【重要】不均衡データ対策
    # 正解データ(System Start)は全体の1/5しかないので、
    # 「見逃したら5倍怒られる」設定にする
    pos_weight = torch.tensor([4.0]).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("--- Training Start ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data) # Logits
            
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 推論時はSigmoidを通して判定
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).float()
            correct += (predicted == target).sum().item()
            total_samples += target.size(0)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            acc = 100. * correct / total_samples
            print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

    # === 検証フェーズ ===
    print("\n--- Final Validation ---")
    model.eval()
    
    test_words = ["System Start", "Music Start", "System Stop", "Hello Gemini"]
    
    for word in test_words:
        filename = f"generated_audio/{word.replace(' ', '_')}.mp3"
        y, sr = librosa.load(filename, sr=16000)
        
        # 簡易ノイズ
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
    
    analyze_internal_state(model, train_dataset)

def analyze_internal_state(model, dataset):
    print("\n--- Internal Reaction Analysis ---")
    words = ["System Start", "Music Start", "System Stop", "Hello Gemini"]
    
    for word in words:
        filename = f"generated_audio/{word.replace(' ', '_')}.mp3"
        y, sr = librosa.load(filename, sr=16000)
        feature = dataset.extract_segmented_feature(y, sr)
        tensor_in = torch.from_numpy(feature).unsqueeze(0).to("cpu") # CPUで計算
        
        # モデルの内部にアクセス
        model.eval()
        with torch.no_grad():
            # Phase 1
            encoded = model.encoder(tensor_in)
            x_curr = F.leaky_relu(encoded, negative_slope=0.01)
            x_curr = torch.abs(x_curr)
            sum_x = torch.sum(x_curr, dim=1, keepdim=True)
            x_curr = x_curr / (sum_x + 1e-8)
            
            # Phase 2 (Reaction Loop)
            max_reaction = 0.0
            for _ in range(model.steps):
                x_mixed = (1 - model.decay) * x_curr
                interaction = torch.einsum('ijk,bi,bj->bk', model.W, x_mixed, x_mixed)
                
                # ここで「反応の激しさ」を計測
                reaction_magnitude = torch.norm(interaction).item()
                if reaction_magnitude > max_reaction:
                    max_reaction = reaction_magnitude
                    
                x_pre = x_mixed + model.alpha * interaction
                x_act = F.leaky_relu(x_pre, negative_slope=0.01)
                x_act = torch.abs(x_act)
                sum_x = torch.sum(x_act, dim=1, keepdim=True)
                x_curr = x_act / (sum_x + 1e-8)
            
            # 最終出力
            final_prob = torch.sigmoid(model.readout(x_curr)).item()
            
        print(f"Input: '{word}'")
        print(f"  -> Max Reaction Power: {max_reaction:.4f} (Chemical Activity)")
        print(f"  -> Final Probability : {final_prob:.4f}")

if __name__ == "__main__":
    train_segmented_audio()