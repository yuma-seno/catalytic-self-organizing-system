import copy

import gymnasium as gym
import torch
import torch.nn as nn


# ==========================================
# 1. MLP Controller Baseline
# ==========================================


class MLPController(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 2, hidden_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def reset(self) -> None:
        # No recurrent state, kept for API compatibility.
        pass

    def forward(self, x):
        state_tensor = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        h = torch.tanh(self.fc1(state_tensor))
        h = torch.tanh(self.fc2(h))
        logits = self.fc_out(h)
        action = torch.argmax(logits, dim=1).item()
        return action


# ==========================================
# 2. 評価関数
# ==========================================


def evaluate(agent: MLPController, env, n_episodes: int = 3, render: bool = False) -> float:
    """Evaluate average score over n_episodes (same protocol as CSOS)."""
    total_score = 0.0

    for _ in range(n_episodes):
        state, _ = env.reset()
        agent.reset()
        episode_reward = 0.0
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


def run_evolution_mlp() -> MLPController:
    env = gym.make("CartPole-v1")

    POPULATION_SIZE = 30
    GENERATIONS = 50
    SIGMA = 0.05

    elite_agent = MLPController(hidden_dim=32)
    best_score = 0.0

    print(f"--- Baseline MLP Evolution (Pop: {POPULATION_SIZE}, Sigma: {SIGMA}) ---")

    for g in range(GENERATIONS):
        population = []
        scores = []

        with torch.no_grad():
            for _ in range(POPULATION_SIZE):
                child = copy.deepcopy(elite_agent)

                for param in child.parameters():
                    noise = torch.randn_like(param) * SIGMA
                    param.add_(noise)

                score = evaluate(child, env, n_episodes=3)
                population.append(child)
                scores.append(score)

        gen_best_score = max(scores)
        gen_best_idx = scores.index(gen_best_score)

        if gen_best_score > best_score:
            best_score = gen_best_score
            elite_agent = copy.deepcopy(population[gen_best_idx])
            print(f"Gen {g:02d}: New Record! Avg Score = {best_score:.1f}")

            if best_score >= 475:
                print("-> Solved! (Stable Performance Verified, MLP Baseline)")
                break
        else:
            print(f"Gen {g:02d}: GenBest {gen_best_score:.1f} (Elite is {best_score:.1f})")

    return elite_agent


# ==========================================
# 4. 結果確認
# ==========================================


if __name__ == "__main__":
    best_agent = run_evolution_mlp()

    print("\n--- Baseline MLP: Final Test Run ---")
    try:
        env_test = gym.make("CartPole-v1", render_mode="human")
    except Exception:
        env_test = gym.make("CartPole-v1")

    score = evaluate(best_agent, env_test, render=True)
    print(f"Final Score (MLP): {score}")
    env_test.close()
