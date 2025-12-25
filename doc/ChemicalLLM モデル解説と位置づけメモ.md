# ChemicalLLM モデル解説と位置づけメモ

## 1. モデルの目的と全体像

- **タスク**:
  - ごく小さなおもちゃデータ上での「次単語予測」（言語モデル）
  - 例: `"cat eat fish ."` から `"eat"` や `"fish"`、 `"."` などを予測する

- **特徴**:
  - 一般的な RNN / LSTM / Transformer ではなく、
  - **「分子濃度 + 化学反応ダイナミクス」**で内部状態を更新する RNN 風モデル

- **基本アイデア**:
  1. 単語 ID を `num_basis` 次元のベクトル（「分子濃度」）に埋め込む（Embedding）
  2. 3階テンソル `W` による二次相互作用（化学反応）で内部状態 `hidden` を更新
  3. 更新された `hidden` から次単語の分布を `decoder` で出力

---

## 2. モデル構造の詳細

### 2.1 パラメータ

```python
class ChemicalLLM(nn.Module):
    def __init__(self, vocab_size, num_basis=32, decay=0.1, alpha=0.2):
        self.N = num_basis     # 分子の種類数 / 内部状態の次元
        self.decay = decay     # 前状態の減衰率
        self.alpha = alpha     # 反応項の強さ

        self.embedding = nn.Embedding(vocab_size, num_basis)
        self.W = nn.Parameter(torch.randn(N, N, N) * 0.05)  # 化学反応テンソル
        self.decoder = nn.Linear(num_basis, vocab_size)     # hidden -> 単語分布
```

- `embedding`:
  - 単語 ID → `num_basis` 次元の連続ベクトル
  - 出力に ReLU をかけて**非負の「分子濃度」**として扱う

- `W`（3階テンソル）:
  - 形状 `(N, N, N)`  
  - `W[k, i, j]` が「状態成分 i, j が反応して k を生成/変化させる」強さを表すイメージ

- `decoder`:
  - 最終的な hidden を語彙サイズに投影し、各単語のロジット（スコア）を出す

---

### 2.2 時系列方向の更新（forward）

```python
def forward(self, input_ids, hidden=None):
    batch_size, seq_len = input_ids.shape
    if hidden is None:
        hidden = torch.zeros(batch_size, self.N, device=input_ids.device)

    outputs = []
    for t in range(seq_len):
        word_input = torch.relu(self.embedding(input_ids[:, t]))

        # 1) Mixing: 前状態 + 現在単語の入力
        x_mixed = (1 - self.decay) * hidden + word_input

        # 2) Reaction: 2体反応による生成 / 変化
        interaction = torch.einsum('ijk,bi,bj->bk', self.W, x_mixed, x_mixed)

        # 3) Update: 元の混合状態 + 反応項
        x_pre = x_mixed + self.alpha * interaction
        x_activated = torch.relu(x_pre)

        # 4) Normalize: 非負 + 総和=1 の分布に正規化
        sum_x = torch.sum(x_activated, dim=1, keepdim=True)
        hidden = x_activated / (sum_x + 1e-8)

        # 5) 出力ロジットを保存
        outputs.append(self.decoder(hidden).unsqueeze(1))

    return torch.cat(outputs, dim=1), hidden
```

#### ステップごとの役割

1. **Embedding + ReLU (`word_input`)**
   - 各単語を非負の「入力濃度ベクトル」に変換

2. **Mixing (`x_mixed`)**
   - 前の状態 `hidden`（前までの文脈）に減衰 `(1 - decay)` をかけ、
   - そこに現在単語の濃度 `word_input` を加える  
   → 「前までの文脈」と「現在単語入力」が混ざった状態

3. **Reaction (`interaction`)**
   - einsum による二次相互作用:
     \[
       \text{interaction}_k = \sum_{i,j} W_{kij} \, x_i \, x_j
     \]
   - 分子 i, j の組み合わせで k が生成される「化学反応」のような動き

4. **Update + ReLU + Normalize**
   - `x_pre = x_mixed + alpha * interaction`
   - ReLU により**負の濃度を排除**
   - 総和で割って**常に「非負 + 総和=1」な確率/濃度ベクトル**を維持
   - これが次ステップの `hidden` になる（RNN の状態更新に相当）

5. **Decoder**
   - 各時刻の `hidden` から次単語のロジットを出力  
   → CrossEntropyLoss で「次の単語」を学習

---

## 3. データセットと学習

### 3.1 おもちゃデータ

```python
idx2word = {0: '.', 1: 'cat', 2: 'dog', 3: 'bird', 4: 'fish',
            5: 'eat', 6: 'fly', 7: 'swim', 8: 'meat', 9: 'sky', 10: 'sea'}

sentences = [
    ["cat",  "eat", "fish", "."],
    ["dog",  "eat", "meat", "."],
    ["bird", "fly", "sky",  "."],
    ["fish", "swim","sea",  "."],
    ["cat",  "eat", "meat", "."]
]
```

- ごく少数のパターン:
  - 「動物 + 動詞 + 食べ物/場所 + `.`」という形
- 目的:
  - ChemicalLLM が簡単な**構文・意味パターン**を内部状態（分子濃度のダイナミクス）として学べるかを見るための**おもちゃ実験**

### 3.2 学習ループ

```python
model = ChemicalLLM(vocab_size=VOCAB_SIZE, num_basis=32)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(501):
    optimizer.zero_grad()
    inputs = train_tensor[:, :-1]   # 先頭〜最後-1
    targets = train_tensor[:, 1:]   # 1つ右にシフト

    outputs, _ = model(inputs)
    loss = criterion(outputs.reshape(-1, VOCAB_SIZE),
                     targets.reshape(-1))
    loss.backward()
    optimizer.step()
```

- 教師あり言語モデル:
  - 入力: `[w_0, w_1, w_2]`
  - 目的: `[w_1, w_2, w_3]` を当てる
- 各ステップの `decoder(hidden)` の出力をまとめて `CrossEntropyLoss` にかける

---

## 4. 生成テスト（推論）

```python
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
```

- 開始単語を与え、その都度
  1. ChemicalLLM に単語を1つだけ入れる
  2. 出てきた分布の argmax で次単語を選ぶ
  3. 選んだ単語を次の入力として再帰的に生成
- 例:
  - `"bird"` → 学習がうまくいくと `"fly sky ."` のような系列を期待

---

## 5. このモデルに対する所見（意見）

### 5.1 おもしろい点

- **RNN 的構造を化学反応のメタファーで書き直したモデル**
  - 通常の RNN は `hidden = f(hidden, input)` として黒箱で更新される
  - このモデルでは、更新が
    - 線形和（Mixing）
    - 3階テンソルによる二次相互作用（Reaction）
    - 非負制約 + 正規化（確率/濃度として解釈）
    に分解されており、**何が起こっているかの物理・化学的イメージがつきやすい**

- **状態が常に「非負 + 総和=1」**
  - 内部状態を「分子濃度」「分布」として一貫した解釈ができる
  - CSOS_MNIST と同様、**確率単体上のダイナミカルシステム**になっている

- **小さなおもちゃデータで挙動を直感的に観察できる**
  - `num_basis`, `decay`, `alpha` をいじりつつ、
    - 文脈の保持能力
    - 単語パターンの一般化
  を手軽に試せる

### 5.2 課題・注意点

- **パラメータ数（O(N^3)）のスケール**
  - `num_basis=32` なら問題ないが、スケールを上げるとすぐ大きくなる
  - 大規模 LLM にこのまま使うのは難しいため、
    - テンソル分解
    - 構造化テンソル（例: 低ランク + 対称性など）
    の導入が必要になりそう

- **安定性ハイパーパラメータ (`decay`, `alpha`) 依存**
  - `alpha` を下げて安定化させているコメントがある通り、
  - 反応が強すぎると hidden が暴走しやすい
  - 逆に弱すぎると「ほぼ線形 + ReLU + 正規化」のつまらないモデルになる可能性もある

---

## 6. 類似・関連する研究の方向性

この ChemicalLLM も、以下のような研究領域と思想的に近い。

1. **ダイナミカルシステムとしての RNN / 言語モデル**
   - `"recurrent neural network dynamical system"`
   - `"neural ODE sequence modeling"`
   - hidden を微分方程式や力学系として扱う研究

2. **高次相互作用 RNN / Multiplicative RNN**
   - `"multiplicative RNN"`
   - `"higher-order recurrent neural network"`
   - 埋め込み同士の積 `x_i x_j` を明示的に扱う RNN は古くから提案されており、今回の 3階テンソル更新とよく似ている

3. **エネルギーベースモデル / Hopfield 的なシーケンスモデル**
   - `"energy-based model for sequences"`
   - `"Hopfield network for sequence modeling"`
   - 反復更新・安定点・記憶という観点からの類似性

4. **化学反応ネットワーク（CRN）×機械学習**
   - `"chemical reaction network learning"`
   - `"neural network as chemical reaction"`
   - 実際の CRN を用いて計算・学習させる試み / CRN でNNを近似する研究など

---

## 7. まとめ

- **何をしているか**:
  - 小さな語彙と短い文からなるおもちゃコーパスに対して、
  - 内部状態を「分子濃度ベクトル」とし、
  - 3階テンソルで定義された化学反応ダイナミクスを用いて
  - RNN 的に文脈を更新しながら次単語を予測する言語モデル。

- **意義**:
  - 通常の RNN / LSTM / Transformer と違う観点から、
    - 「意味」や「構文」が**どのような内部ダイナミクスとして表現されうるか**を探る実験プラットフォームになりうる。
  - CSOS_MNIST と合わせて、
    - 「化学反応ダイナミクスで知能を記述する」という一貫した枠組みを作ろうとしている点が興味深い。