import torch
import torch.nn as nn
import numpy as np
import copy
import gymnasium as gym

# ==========================================
# 1. CSOS Controller Model
# ==========================================
class CSOSController(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, num_basis=32, decay=0.1):
        super().__init__()
        self.N = num_basis
        self.decay = decay
        self.input_dim = input_dim
        
        self.encoder = nn.Linear(input_dim, num_basis) 
        self.W = nn.Parameter(torch.randn(num_basis, num_basis, num_basis) * 0.01)
        self.decoder = nn.Linear(num_basis, output_dim)
        
        # 初期状態
        self.state = torch.zeros(1, num_basis)

    def reset(self):
        # リセット時も勾配を切る
        self.state = torch.zeros(1, self.N)

    def forward(self, x):
        # 入力変換
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        encoded = torch.relu(self.encoder(x_tensor))
        
        # 1. Mixing
        x_mixed = (1 - self.decay) * self.state + encoded
        
        # 2. Reaction
        interaction = torch.einsum('ijk,bi,bj->bk', self.W, x_mixed, x_mixed)
        
        # 3. Update
        x_pre = x_mixed + 0.5 * interaction
        new_state = torch.relu(x_pre)
        
        # 4. Normalize
        sum_x = torch.sum(new_state, dim=1, keepdim=True)
        new_state = new_state / (sum_x + 1e-8)
        
        # 【修正点】計算履歴を断ち切る (これをしないとdeepcopyでエラーになる)
        self.state = new_state.detach()
        
        # 5. Action Decision
        logits = self.decoder(self.state)
        action = torch.argmax(logits, dim=1).item()
        return action

# ==========================================
# 2. 評価関数
# ==========================================
def evaluate(agent, env, n_episodes=3, render=False):
    """
    n_episodes回実行して、その平均スコアを返す（まぐれ防止）
    """
    total_score = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        agent.reset()
        episode_reward = 0
        truncated = False
        terminated = False
        
        while not (terminated or truncated):
            if render:
                env.render()
                
            action = agent(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if episode_reward >= 500:
                break
        
        total_score += episode_reward
            
    return total_score / n_episodes

# ==========================================
# 3. 進化戦略
# ==========================================
def run_evolution():
    env = gym.make("CartPole-v1")
    
    POPULATION_SIZE = 30
    GENERATIONS = 50      # 増やす: 20 -> 50 (安定して伸びるのを待つ)
    SIGMA = 0.05          # 減らす: 0.1 -> 0.05 (破壊的な変異を防ぐ)
    
    elite_agent = CSOSController(num_basis=16)
    best_score = 0
    
    print(f"--- Start Robust Evolution (Pop: {POPULATION_SIZE}, Sigma: {SIGMA}) ---")
    
    for g in range(GENERATIONS):
        population = []
        scores = []
        
        with torch.no_grad():
            for _ in range(POPULATION_SIZE):
                child = copy.deepcopy(elite_agent)
                
                for param in child.parameters():
                    # 変異を少しマイルドに
                    noise = torch.randn_like(param) * SIGMA
                    param.add_(noise)
                
                # 3回平均で評価
                score = evaluate(child, env, n_episodes=3)
                population.append(child)
                scores.append(score)
        
        gen_best_score = max(scores)
        gen_best_idx = scores.index(gen_best_score)
        
        # エリート更新判定
        if gen_best_score > best_score:
            best_score = gen_best_score
            elite_agent = copy.deepcopy(population[gen_best_idx])
            print(f"Gen {g:02d}: New Record! Avg Score = {best_score:.1f}")
            
            # 平均475を超えれば、ほぼ間違いなく本物
            if best_score >= 475:
                print("-> Solved! (Stable Performance Verified)")
                break
        else:
            # 進捗が見えないと不安なので、世代の最高得点も表示
            print(f"Gen {g:02d}: GenBest {gen_best_score:.1f} (Elite is {best_score:.1f})")

    return elite_agent

# ==========================================
# 4. 結果確認
# ==========================================
if __name__ == "__main__":
    best_agent = run_evolution()
    
    print("\n--- Final Test Run ---")
    try:
        # 可能な環境なら描画モードで
        env_test = gym.make("CartPole-v1", render_mode="human")
    except:
        env_test = gym.make("CartPole-v1")

    score = evaluate(best_agent, env_test, render=True)
    print(f"Final Score: {score}")
    env_test.close()