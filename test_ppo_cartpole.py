#!/usr/bin/env python3
"""
CartPole-v1 Validation Test for PPO
SUCCESS CRITERIA: Average episode length ≥ 475 over last 100 episodes
EXPECTED: Solve in <1000 updates (5-10 minutes)
"""

import gymnasium as gym
import numpy as np
import time
from ppo_simple import SimplePPOAgent, SimplePPOConfig

def test_cartpole(max_updates=1000, rollout_steps=2048):
    """Test PPO on CartPole-v1"""

    print("="*80, flush=True)
    print("PPO CARTPOLE VALIDATION TEST", flush=True)
    print("="*80, flush=True)
    print(f"Success Criterion: Episode length ≥ 475 (averaged over last 100 episodes)", flush=True)
    print(f"Max Updates: {max_updates}", flush=True)
    print(f"Rollout Steps: {rollout_steps}", flush=True)
    print("="*80, flush=True)

    # Create environment
    env = gym.make('CartPole-v1')

    # Create agent
    config = SimplePPOConfig(
        hidden_size=64,
        learning_rate=3e-4,
        ppo_epochs=4,
        minibatch_size=64,
        clip_epsilon=0.2,
        entropy_coef=0.01,
    )

    agent = SimplePPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        continuous=False,
        config=config
    )

    # Training metrics
    episode_lengths = []
    episode_rewards = []
    update_times = []

    start_time = time.time()
    solved = False
    solved_at_update = None

    print("\nStarting training...", flush=True)
    print("-"*80, flush=True)

    for update in range(max_updates):
        update_start = time.time()

        # Collect rollout
        obs, _ = env.reset()
        episode_length = 0
        episode_reward = 0
        steps = 0

        while steps < rollout_steps:
            action, log_prob, value = agent.get_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, value, log_prob, done)

            episode_length += 1
            episode_reward += reward
            steps += 1

            if done:
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)

                obs, _ = env.reset()
                episode_length = 0
                episode_reward = 0
            else:
                obs = next_obs

        # PPO update
        metrics = agent.update(obs, done)

        update_time = time.time() - update_start
        update_times.append(update_time)

        # Check if solved
        if len(episode_lengths) >= 100:
            recent_avg_length = np.mean(episode_lengths[-100:])

            if recent_avg_length >= 475 and not solved:
                solved = True
                solved_at_update = update + 1
                print(f"\n{'='*80}", flush=True)
                print(f"SOLVED at update {solved_at_update}!", flush=True)
                print(f"   Average episode length: {recent_avg_length:.1f} (target: ≥475)", flush=True)
                print(f"   Time taken: {time.time() - start_time:.1f}s", flush=True)
                print(f"{'='*80}", flush=True)
                break

        # Progress reporting
        if (update + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            recent_lengths = episode_lengths[-10:] if len(episode_lengths) >= 10 else episode_lengths
            avg_length_100 = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else 0

            print(f"Update {update+1:4d} | "
                  f"Episodes: {len(episode_lengths):4d} | "
                  f"Reward (last 10): {np.mean(recent_rewards):7.1f} | "
                  f"Length (last 10): {np.mean(recent_lengths):6.1f} | "
                  f"Length (last 100): {avg_length_100:6.1f} | "
                  f"VLoss: {metrics.get('value_loss', 0):.3f} | "
                  f"Entropy: {metrics.get('entropy', 0):.3f}", flush=True)

    env.close()

    # Final report
    print("\n" + "="*80, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*80, flush=True)

    total_time = time.time() - start_time

    if solved:
        print(f"TEST PASSED", flush=True)
        print(f"   Solved at update: {solved_at_update}/{max_updates}", flush=True)
        print(f"   Final avg episode length (last 100): {np.mean(episode_lengths[-100:]):.1f}", flush=True)
        print(f"   Total time: {total_time:.1f}s", flush=True)
        print(f"   Updates per second: {agent.total_updates / total_time:.2f}", flush=True)
    else:
        if len(episode_lengths) >= 100:
            final_avg = np.mean(episode_lengths[-100:])
            print(f"TEST FAILED", flush=True)
            print(f"   Did not solve in {max_updates} updates", flush=True)
            print(f"   Final avg episode length (last 100): {final_avg:.1f} (target: ≥475)", flush=True)
            print(f"   Progress: {final_avg / 475 * 100:.1f}%", flush=True)
        else:
            print(f"TEST FAILED", flush=True)
            print(f"   Not enough episodes ({len(episode_lengths)}) to evaluate", flush=True)

    print(f"   Total episodes: {len(episode_lengths)}", flush=True)
    print(f"   Total updates: {agent.total_updates}", flush=True)
    print(f"   Total time: {total_time:.1f}s", flush=True)

    if len(update_times) > 0:
        print(f"   Avg update time: {np.mean(update_times):.3f}s", flush=True)

    print("="*80, flush=True)

    return solved, episode_lengths, episode_rewards

if __name__ == "__main__":
    solved, lengths, rewards = test_cartpole(max_updates=1000)

    if solved:
        print("\nPPO implementation is WORKING!", flush=True)
        print("   Next step: Test on Pendulum-v1 (continuous actions)", flush=True)
    else:
        print("\nPPO implementation has ISSUES", flush=True)
        print("   Need to debug before proceeding to trading", flush=True)
