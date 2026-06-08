#!/usr/bin/env python3
"""
Analyze the training performance from job 2051893429931081728 to understand what the agent learned.
This analyzes the training logs to extract learning patterns and identify potential issues.
"""
import pandas as pd
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_training_logs():
    """Parse the training log data from job 2051893429931081728"""
    
    # Training data extracted from the logs you provided
    training_data = [
        # Format: (update, reward, policy_loss, value_loss, entropy, episodes, ep_len)
        (10, -0.8331, 0.3612, 162.7766, 1.4253, 45, 5.6),
        (20, 0.2764, 0.2877, 229.0583, 1.4315, 42, 6.1),
        (30, 0.2033, 6.0211, 200.3172, 1.4339, 49, 5.2),
        (40, 0.3288, 0.4226, 277.3105, 1.4380, 48, 5.3),
        (50, 0.0711, 0.3754, 269.8900, 1.4419, 43, 5.9),
        (60, -0.5931, 0.2272, 245.5234, 1.4455, 52, 4.7),
        (70, -0.7417, 0.6294, 215.5717, 1.4504, 49, 5.2),
        (80, 0.2101, 0.3347, 281.0934, 1.4554, 43, 5.9),
        (90, 0.4449, 0.3746, 290.5025, 1.4596, 48, 5.3),
        (100, 0.4206, 0.1829, 1980.9800, 1.4637, 42, 6.1),
        (110, -0.0379, 0.6641, 209.1746, 1.4683, 39, 6.6),
        (120, -0.2828, 0.4280, 203.9038, 1.4750, 50, 5.1),
        (130, 0.0040, 0.4142, 311.6376, 1.4821, 48, 5.3),
        (140, -0.0819, 0.6705, 239.6140, 1.4868, 48, 5.3),
        (150, -0.9491, 0.6151, 246.8904, 1.4927, 53, 4.8),
        (160, -0.3702, 0.2679, 261.2233, 1.5001, 51, 5.0),
        (170, 0.0002, 1.1039, 238.4730, 1.5071, 49, 5.2),
        (180, 0.3416, 0.3644, 343.6426, 1.5120, 45, 5.7),
        (190, -0.0196, 0.3109, 273.2599, 1.5178, 51, 4.9),
        (200, -0.3334, 1.0349, 168.4703, 1.5258, 47, 5.4),
        (210, 0.1373, 0.5038, 198.8837, 1.5332, 49, 5.2),
        (220, -0.0672, 0.3801, 322.9373, 1.5446, 53, 4.8),
        (230, -0.2742, 0.4421, 218.0872, 1.5504, 47, 5.3),
        (240, 0.2792, 1.2275, 245.5649, 1.5617, 42, 6.0),
        (250, -0.0062, 2.5213, 154.4751, 1.5705, 45, 5.7),
        (260, -0.9389, 0.4564, 130.9245, 1.5794, 57, 4.5),
        (270, 0.2313, 0.3291, 319.5552, 1.5849, 48, 5.3),
        (280, -0.0500, 1.6490, 182.3737, 1.5896, 46, 5.4),
        (290, 0.3844, 0.3129, 368.3302, 1.6040, 46, 5.5),
        (300, 0.4505, 0.4395, 209.6284, 1.6101, 47, 5.4),
        (310, -0.8474, 0.3036, 157.6346, 1.6176, 54, 4.7),
        (320, 0.0250, 0.3315, 173.7739, 1.6210, 50, 5.0),
        (330, -0.0285, 0.5838, 262.9105, 1.6261, 52, 4.9),
        (340, -0.0916, 0.9870, 6531.6206, 1.6374, 45, 5.7),
        (350, -0.2146, 0.2421, 415.7608, 1.6430, 54, 4.7),
        (360, -0.4845, 0.7311, 134.3119, 1.6493, 60, 4.2),
        (370, -0.1500, 0.3469, 168.3566, 1.6545, 46, 5.4),
        (380, 0.0217, 0.7078, 249.7179, 1.6624, 45, 5.4),
        (390, 0.7874, 0.4324, 346.2441, 1.6700, 48, 5.0),
        (400, 0.0936, 0.9627, 219.4643, 1.6763, 55, 4.6),
        (410, -1.1585, 0.4180, 169.2283, 1.6836, 58, 4.4),
        (420, 0.3194, 0.4652, 264.8117, 1.6894, 45, 5.7),
        (430, 0.5065, 1.0379, 347.7534, 1.7043, 51, 4.9),
        (440, -0.3645, 0.5619, 199.7835, 1.7097, 50, 5.1),
        (450, 0.2503, 0.4968, 351.0122, 1.7176, 48, 5.3),
        (460, 0.2357, 0.3818, 204.0855, 1.7291, 46, 5.5),
        (470, 0.2906, 0.7435, 198.8303, 1.7365, 52, 4.9),
        (480, 0.0674, 0.3575, 171.4102, 1.7442, 56, 4.5),
        (490, 0.4024, 0.3126, 257.6261, 1.7503, 43, 5.9),
        (500, -0.7047, 7.2069, 244.0829, 1.7640, 60, 4.2),
        (510, 0.5812, 0.5588, 302.6469, 1.7743, 43, 5.8),
        (520, -0.0234, 0.4326, 258.5136, 1.7806, 48, 5.1),
        (530, 0.3810, 0.4712, 302.5099, 1.7872, 46, 5.5),
        (540, -0.1519, 0.4914, 215.5098, 1.7932, 50, 5.1),
        (550, 0.6642, 0.3208, 1669.4965, 1.7982, 47, 5.4),
        (560, 0.3601, 0.2854, 215.3755, 1.8053, 49, 5.1),
        (570, -0.6940, 0.4295, 165.8869, 1.8139, 59, 4.3),
        (580, -0.3117, 1.1782, 178.5188, 1.8203, 54, 4.7),
        (590, -0.1796, 0.8959, 135.8665, 1.8278, 50, 5.1),
        (600, 0.5862, 0.3888, 228.3903, 1.8328, 46, 5.5),
        (610, 0.2753, 0.9212, 301.6263, 1.8380, 48, 5.3),
        (620, -0.1282, 0.5546, 169.1845, 1.8421, 56, 4.4),
        (630, -0.5028, 0.3413, 244.8140, 1.8475, 54, 4.7),
        (640, 0.0133, 2.0973, 247.6097, 1.8529, 56, 4.6),
        (650, -0.1893, 0.4468, 219.6600, 1.8589, 55, 4.6),
        (660, 0.0016, 0.8820, 184.6974, 1.8652, 52, 4.9),
        (670, 0.1443, 0.3422, 247.3348, 1.8709, 59, 4.3),
        (680, -0.7660, 0.4929, 160.9675, 1.8769, 53, 4.8),
        (690, 0.3913, 0.3306, 295.5459, 1.8936, 51, 5.0),
        (700, 0.5168, 0.6057, 214.5294, 1.8991, 49, 5.2),
        (710, -0.3643, 0.4659, 211.7104, 1.9139, 57, 4.5),
        (720, -0.3681, 0.9510, 138.1952, 1.9193, 54, 4.6),
        (730, -0.6409, 0.2794, 135.2213, 1.9242, 57, 4.5),
        (740, 0.2496, 0.7897, 2277.6338, 1.9280, 50, 5.1),
        (750, 0.5305, 0.3298, 229.3158, 1.9307, 55, 4.6),
        (760, 0.5542, 0.8435, 371.9780, 1.9332, 47, 5.4),
        (770, -0.9512, 0.8954, 484.1205, 1.9400, 60, 4.2),
        (780, -1.3337, 0.2781, 125.3951, 1.9486, 62, 4.1),
        (790, -0.9435, 0.6791, 134.2104, 1.9620, 63, 4.0),
        (800, -0.0262, 0.4538, 179.2420, 1.9681, 53, 4.7),
        (810, -0.0660, 1.3436, 171.6242, 1.9789, 53, 4.7),
        (820, -0.3170, 0.2354, 180.2882, 1.9852, 53, 4.6),
        (830, -0.1856, 0.7014, 158.8766, 1.9885, 59, 4.3),
        (840, -0.6838, 0.4977, 121.4050, 1.9935, 64, 3.9),
        (850, -0.9569, 0.9832, 173.0897, 1.9971, 63, 4.0),
        (860, -0.3405, 1.1055, 159.5787, 2.0036, 51, 5.0),
        (870, -0.0376, 1.3175, 183.4464, 2.0083, 51, 5.0),
        (880, 0.0092, 0.3578, 194.9963, 2.0125, 54, 4.7),
        (890, 0.1699, 0.1608, 722305.6875, 2.0169, 46, 5.3),
        (900, -0.7792, 0.2824, 142.1526, 2.0207, 64, 4.0),
        (910, -0.2588, 0.7319, 174.1524, 2.0276, 56, 4.5),
        (920, -0.2344, 0.4464, 209.7870, 2.0313, 53, 4.8),
        (930, -0.0223, 0.3244, 318.5580, 2.0368, 50, 4.9),
        (940, -0.6174, 0.3759, 203.7104, 2.0431, 64, 4.0),
        (950, -0.4621, 0.4445, 203.9071, 2.0479, 59, 4.3),
        (960, 0.0792, 1.3733, 228.7674, 2.0527, 51, 5.0),
        (970, -0.4312, 0.5005, 158.1324, 2.0599, 55, 4.6),
        (980, 0.0134, 0.3496, 198.0324, 2.0639, 54, 4.7),
        (990, 0.1951, 0.3828, 186.2192, 2.0669, 51, 5.0),
        (1000, 0.4138, 0.3739, 202.4015, 2.0718, 53, 4.8),
        (1010, 0.0764, 0.2573, 205.0266, 2.0768, 57, 4.5),
        (1020, -0.4647, 0.3257, 175.0340, 2.0818, 56, 4.5),
    ]
    
    return pd.DataFrame(training_data, columns=[
        'update', 'reward', 'policy_loss', 'value_loss', 'entropy', 'episodes', 'ep_len'
    ])

def analyze_learning_progression(df):
    """Analyze the learning progression to understand what the agent learned"""
    
    logger.info("🔍 ANALYZING TRAINING PROGRESSION")
    logger.info("=================================")
    
    # Basic statistics
    logger.info(f"Training Updates: {df['update'].min()} to {df['update'].max()}")
    logger.info(f"Total Training Points: {len(df)}")
    
    # Reward analysis
    initial_reward = df['reward'].iloc[0]
    final_reward = df['reward'].iloc[-1]
    peak_reward = df['reward'].max()
    worst_reward = df['reward'].min()
    mean_reward = df['reward'].mean()
    
    logger.info("\n📈 REWARD PROGRESSION:")
    logger.info(f"  Initial Reward (Update 10): {initial_reward:.4f}")
    logger.info(f"  Final Reward (Update 1020): {final_reward:.4f}")
    logger.info(f"  Peak Reward: {peak_reward:.4f}")
    logger.info(f"  Worst Reward: {worst_reward:.4f}")
    logger.info(f"  Mean Reward: {mean_reward:.4f}")
    logger.info(f"  Improvement: {final_reward - initial_reward:+.4f}")
    
    # Find best performance periods
    df['reward_ma'] = df['reward'].rolling(10, min_periods=1).mean()
    best_period_idx = df['reward_ma'].idxmax()
    best_period = df.loc[max(0, best_period_idx-5):best_period_idx+5]
    
    logger.info("\n🏆 BEST PERFORMANCE PERIOD:")
    logger.info(f"  Around Update: {df.loc[best_period_idx, 'update']}")
    logger.info(f"  Best 10-update average: {df.loc[best_period_idx, 'reward_ma']:.4f}")
    logger.info(f"  Updates {best_period['update'].min()}-{best_period['update'].max()}: {best_period['reward'].tolist()}")
    
    # Analyze learning phases
    early_phase = df[df['update'] <= 200]['reward'].mean()
    mid_phase = df[(df['update'] > 200) & (df['update'] <= 600)]['reward'].mean() 
    late_phase = df[df['update'] > 600]['reward'].mean()
    
    logger.info("\n📊 LEARNING PHASES:")
    logger.info(f"  Early Phase (Updates 10-200): {early_phase:.4f} avg reward")
    logger.info(f"  Mid Phase (Updates 200-600): {mid_phase:.4f} avg reward")
    logger.info(f"  Late Phase (Updates 600-1020): {late_phase:.4f} avg reward")
    
    # Positive vs negative rewards
    positive_rewards = df[df['reward'] > 0]
    negative_rewards = df[df['reward'] < 0]
    
    logger.info("\n✅ POSITIVE PERFORMANCE:")
    logger.info(f"  Positive Reward Updates: {len(positive_rewards)}/{len(df)} ({len(positive_rewards)/len(df)*100:.1f}%)")
    logger.info(f"  Average Positive Reward: {positive_rewards['reward'].mean():.4f}")
    logger.info(f"  Best Positive Streaks: {find_streaks(df['reward'] > 0)}")
    
    logger.info("\n❌ NEGATIVE PERFORMANCE:")
    logger.info(f"  Negative Reward Updates: {len(negative_rewards)}/{len(df)} ({len(negative_rewards)/len(df)*100:.1f}%)")
    logger.info(f"  Average Negative Reward: {negative_rewards['reward'].mean():.4f}")
    
    # Episode length analysis
    avg_ep_len = df['ep_len'].mean()
    logger.info("\n⏱️ EPISODE ANALYSIS:")
    logger.info(f"  Average Episode Length: {avg_ep_len:.1f} steps")
    logger.info(f"  Episode Length Range: {df['ep_len'].min():.1f} - {df['ep_len'].max():.1f} steps")
    logger.info(f"  Short Episodes (<5 steps): {(df['ep_len'] < 5).sum()} updates")
    logger.info(f"  Long Episodes (>6 steps): {(df['ep_len'] > 6).sum()} updates")
    
    # Entropy analysis (exploration)
    logger.info("\n🔍 EXPLORATION ANALYSIS:")
    logger.info(f"  Initial Entropy: {df['entropy'].iloc[0]:.4f}")
    logger.info(f"  Final Entropy: {df['entropy'].iloc[-1]:.4f}")
    logger.info(f"  Entropy Increase: {df['entropy'].iloc[-1] - df['entropy'].iloc[0]:+.4f}")
    logger.info(f"  Final Entropy Level: {'Good' if df['entropy'].iloc[-1] > 1.8 else 'Low'} exploration")
    
    return df

def find_streaks(boolean_series):
    """Find longest streaks of True values"""
    streaks = []
    current_streak = 0
    
    for val in boolean_series:
        if val:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = 0
    
    if current_streak > 0:
        streaks.append(current_streak)
    
    if not streaks:
        return "None"
    
    return f"Longest: {max(streaks)}, Total streaks: {len(streaks)}"

def identify_issues(df):
    """Identify potential training issues"""
    
    logger.info("\n🚨 POTENTIAL ISSUES ANALYSIS:")
    logger.info("=============================")
    
    issues_found = []
    
    # Check for value loss explosions
    high_value_loss = df[df['value_loss'] > 1000]
    if len(high_value_loss) > 0:
        issues_found.append("Value Loss Explosions")
        logger.info(f"⚠️  Value Loss Explosions: {len(high_value_loss)} updates with loss > 1000")
        logger.info(f"   Max Value Loss: {df['value_loss'].max():.1f} at update {df.loc[df['value_loss'].idxmax(), 'update']}")
    
    # Check for policy loss instability  
    high_policy_loss = df[df['policy_loss'] > 3.0]
    if len(high_policy_loss) > 0:
        issues_found.append("Policy Loss Spikes")
        logger.info(f"⚠️  Policy Loss Spikes: {len(high_policy_loss)} updates with loss > 3.0")
    
    # Check for reward volatility
    reward_std = df['reward'].std()
    if reward_std > 0.5:
        issues_found.append("High Reward Volatility")
        logger.info(f"⚠️  High Reward Volatility: std = {reward_std:.4f}")
    
    # Check for declining performance
    recent_performance = df.tail(20)['reward'].mean()
    peak_performance = df['reward'].rolling(20, min_periods=1).mean().max()
    
    if recent_performance < peak_performance - 0.2:
        issues_found.append("Performance Decline")
        logger.info(f"⚠️  Performance Decline: Recent avg {recent_performance:.4f} vs peak {peak_performance:.4f}")
    
    # Check episode length trends
    if df['ep_len'].mean() < 5.0:
        issues_found.append("Short Episodes")
        logger.info(f"⚠️  Short Episodes: Average {df['ep_len'].mean():.1f} steps (may indicate frequent failures)")
    
    if not issues_found:
        logger.info("✅ No major training issues detected!")
    
    return issues_found

def generate_verdict(df, issues):
    """Generate final verdict on training success"""
    
    logger.info("\n🎯 TRAINING VERDICT:")
    logger.info("===================")
    
    # Scoring criteria
    score = 0
    max_score = 10
    
    # Learning progression (0-3 points)
    improvement = df['reward'].iloc[-1] - df['reward'].iloc[0]
    if improvement > 0.5:
        score += 3
        logger.info("✅ Strong Learning Progression (+3 points)")
    elif improvement > 0:
        score += 2
        logger.info("✅ Moderate Learning Progression (+2 points)")
    elif improvement > -0.2:
        score += 1
        logger.info("⚠️  Weak Learning Progression (+1 point)")
    else:
        logger.info("❌ No Learning Progression (0 points)")
    
    # Peak performance (0-2 points)
    peak_reward = df['reward'].max()
    if peak_reward > 0.6:
        score += 2
        logger.info("✅ Excellent Peak Performance (+2 points)")
    elif peak_reward > 0.3:
        score += 1
        logger.info("✅ Good Peak Performance (+1 point)")
    else:
        logger.info("❌ Low Peak Performance (0 points)")
    
    # Consistency (0-2 points)
    positive_rate = (df['reward'] > 0).mean()
    if positive_rate > 0.6:
        score += 2
        logger.info("✅ High Consistency (+2 points)")
    elif positive_rate > 0.4:
        score += 1
        logger.info("✅ Moderate Consistency (+1 point)")
    else:
        logger.info("❌ Low Consistency (0 points)")
    
    # Stability (0-2 points)
    if len(issues) == 0:
        score += 2
        logger.info("✅ Stable Training (+2 points)")
    elif len(issues) <= 2:
        score += 1
        logger.info("⚠️  Minor Issues (+1 point)")
    else:
        logger.info("❌ Training Issues (0 points)")
    
    # Exploration (0-1 point)
    if df['entropy'].iloc[-1] > 1.8:
        score += 1
        logger.info("✅ Good Exploration (+1 point)")
    else:
        logger.info("⚠️  Limited Exploration (0 points)")
    
    logger.info(f"\n📊 FINAL SCORE: {score}/{max_score}")
    
    if score >= 8:
        verdict = "EXCELLENT - Ready for deployment"
        emoji = "🎉"
    elif score >= 6:
        verdict = "GOOD - Model learned successfully"
        emoji = "✅"
    elif score >= 4:
        verdict = "FAIR - Some learning occurred"
        emoji = "⚠️"
    else:
        verdict = "POOR - Minimal learning"
        emoji = "❌"
    
    logger.info(f"{emoji} VERDICT: {verdict}")
    
    return score, verdict

def main():
    logger.info("🧪 ANALYZING TRAINING PERFORMANCE")
    logger.info("Job ID: 2051893429931081728")
    logger.info("Updates Completed: 1020/3000 (34%)")
    logger.info("=" * 50)
    
    # Load and parse training data
    df = parse_training_logs()
    
    # Analyze learning progression  
    df = analyze_learning_progression(df)
    
    # Identify issues
    issues = identify_issues(df)
    
    # Generate verdict
    score, verdict = generate_verdict(df, issues)
    
    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS COMPLETE")
    
    return score >= 4  # Return True if model learned reasonably well

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)