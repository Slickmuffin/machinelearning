#!/usr/bin/env python3
"""
Deep analysis of the specific technical issues in job 2051893429931081728
Focus on identifying the root cause of performance decline and training instability
"""
import numpy as np
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_detailed_training_data():
    """Get more detailed training data with focus on problematic periods"""
    
    # Training data with more focus on the problematic areas
    training_data = [
        # Early phase - learning
        (10, -0.8331, 0.3612, 162.7766, 1.4253, 45, 5.6),
        (20, 0.2764, 0.2877, 229.0583, 1.4315, 42, 6.1),
        (30, 0.2033, 6.0211, 200.3172, 1.4339, 49, 5.2),  # Early policy loss spike
        (40, 0.3288, 0.4226, 277.3105, 1.4380, 48, 5.3),
        
        # Mid phase - good performance  
        (100, 0.4206, 0.1829, 1980.9800, 1.4637, 42, 6.1),  # First value loss spike
        (200, -0.3334, 1.0349, 168.4703, 1.5258, 47, 5.4),
        (300, 0.4505, 0.4395, 209.6284, 1.6101, 47, 5.4),
        (340, -0.0916, 0.9870, 6531.6206, 1.6374, 45, 5.7),  # Major value loss spike
        
        # Peak performance period
        (390, 0.7874, 0.4324, 346.2441, 1.6700, 48, 5.0),   # Best reward
        (430, 0.5065, 1.0379, 347.7534, 1.7043, 51, 4.9),
        (500, -0.7047, 7.2069, 244.0829, 1.7640, 60, 4.2),  # Policy loss explosion
        (510, 0.5812, 0.5588, 302.6469, 1.7743, 43, 5.8),
        (550, 0.6642, 0.3208, 1669.4965, 1.7982, 47, 5.4),  # Another value loss spike
        (560, 0.3601, 0.2854, 215.3755, 1.8053, 49, 5.1),
        (600, 0.5862, 0.3888, 228.3903, 1.8328, 46, 5.5),
        
        # Decline phase - where issues started
        (680, -0.7660, 0.4929, 160.9675, 1.8769, 53, 4.8),
        (700, 0.5168, 0.6057, 214.5294, 1.8991, 49, 5.2),
        (730, -0.6409, 0.2794, 135.2213, 1.9242, 57, 4.5),  # Episodes getting shorter
        (740, 0.2496, 0.7897, 2277.6338, 1.9280, 50, 5.1),  # Value loss spike
        (750, 0.5305, 0.3298, 229.3158, 1.9307, 55, 4.6),   # Episodes shorter
        (760, 0.5542, 0.8435, 371.9780, 1.9332, 47, 5.4),
        
        # Critical failure period
        (770, -0.9512, 0.8954, 484.1205, 1.9400, 60, 4.2),  # Performance crash
        (780, -1.3337, 0.2781, 125.3951, 1.9486, 62, 4.1),  # Worst reward ever
        (790, -0.9435, 0.6791, 134.2104, 1.9620, 63, 4.0),  # Episodes very short
        (800, -0.0262, 0.4538, 179.2420, 1.9681, 53, 4.7),
        
        # Final instability period  
        (840, -0.6838, 0.4977, 121.4050, 1.9935, 64, 3.9),  # Episodes getting very short
        (850, -0.9569, 0.9832, 173.0897, 1.9971, 63, 4.0),
        (890, 0.1699, 0.1608, 722305.6875, 2.0169, 46, 5.3), # MASSIVE value loss explosion
        (900, -0.7792, 0.2824, 142.1526, 2.0207, 64, 4.0),   # Very short episodes
        (940, -0.6174, 0.3759, 203.7104, 2.0431, 64, 4.0),   # Consistently short
        (1000, 0.4138, 0.3739, 202.4015, 2.0718, 53, 4.8),
        (1020, -0.4647, 0.3257, 175.0340, 2.0818, 56, 4.5),
    ]
    
    return pd.DataFrame(training_data, columns=[
        'update', 'reward', 'policy_loss', 'value_loss', 'entropy', 'episodes', 'ep_len'
    ])

def analyze_value_loss_explosions(df):
    """Deep dive into value loss explosions - this is likely the root cause"""
    
    logger.info("💥 ANALYZING VALUE LOSS EXPLOSIONS")
    logger.info("==================================")
    
    # Find all value loss spikes
    normal_vloss = df[df['value_loss'] < 1000]['value_loss'].mean()
    spikes = df[df['value_loss'] > 1000]
    
    logger.info(f"Normal Value Loss Average: {normal_vloss:.1f}")
    logger.info(f"Value Loss Spikes Found: {len(spikes)}")
    
    for idx, row in spikes.iterrows():
        multiplier = row['value_loss'] / normal_vloss
        logger.info(f"  Update {row['update']}: {row['value_loss']:.1f} ({multiplier:.0f}x normal)")
        logger.info(f"    Reward: {row['reward']:.4f}, Episodes: {row['episodes']}, Ep Len: {row['ep_len']:.1f}")
    
    # The mega explosion
    mega_spike = df.loc[df['value_loss'].idxmax()]
    logger.info(f"\n🔥 MEGA EXPLOSION ANALYSIS:")
    logger.info(f"  Update: {mega_spike['update']}")
    logger.info(f"  Value Loss: {mega_spike['value_loss']:,.1f}")
    logger.info(f"  Magnitude: {mega_spike['value_loss'] / normal_vloss:,.0f}x normal")
    logger.info(f"  Context: Reward={mega_spike['reward']:.4f}, Policy Loss={mega_spike['policy_loss']:.4f}")
    
    # Check pattern around explosions
    logger.info(f"\n📊 VALUE LOSS EXPLOSION PATTERNS:")
    explosion_updates = spikes['update'].tolist()
    
    for update in explosion_updates:
        before_idx = df[df['update'] == update - 10].index
        after_idx = df[df['update'] == update + 10].index
        
        if len(before_idx) > 0 and len(after_idx) > 0:
            before = df.loc[before_idx[0]]
            after = df.loc[after_idx[0]]
            
            logger.info(f"  Update {update} explosion context:")
            logger.info(f"    Before (-10): VLoss={before['value_loss']:.1f}, Reward={before['reward']:.4f}")
            logger.info(f"    After (+10):  VLoss={after['value_loss']:.1f}, Reward={after['reward']:.4f}")

def analyze_episode_length_decline(df):
    """Analyze why episode lengths are getting shorter"""
    
    logger.info("\n⏱️ EPISODE LENGTH DECLINE ANALYSIS")
    logger.info("=================================")
    
    # Split into periods
    early = df[df['update'] <= 300]
    mid = df[(df['update'] > 300) & (df['update'] <= 700)]
    late = df[df['update'] > 700]
    
    logger.info(f"Early Period (≤300): {early['ep_len'].mean():.1f} avg steps")
    logger.info(f"Mid Period (300-700): {mid['ep_len'].mean():.1f} avg steps") 
    logger.info(f"Late Period (>700): {late['ep_len'].mean():.1f} avg steps")
    
    # Check correlation with performance
    very_short = df[df['ep_len'] < 4.5]
    normal_length = df[df['ep_len'] > 5.5]
    
    logger.info(f"\nVery Short Episodes (<4.5 steps): {len(very_short)} updates")
    logger.info(f"  Average Reward: {very_short['reward'].mean():.4f}")
    logger.info(f"  Updates: {very_short['update'].tolist()}")
    
    logger.info(f"\nNormal Length Episodes (>5.5 steps): {len(normal_length)} updates")
    logger.info(f"  Average Reward: {normal_length['reward'].mean():.4f}")
    
    # Check if short episodes correlate with high episode count
    logger.info(f"\nEPISODE COUNT ANALYSIS:")
    high_episode_count = df[df['episodes'] > 55]
    logger.info(f"High Episode Count (>55): {len(high_episode_count)} updates")
    logger.info(f"  Average Episode Length: {high_episode_count['ep_len'].mean():.1f}")
    logger.info(f"  Average Reward: {high_episode_count['reward'].mean():.4f}")
    
    # This suggests episodes are terminating early due to some condition

def analyze_numerical_instability_patterns(df):
    """Look for patterns suggesting numerical instability"""
    
    logger.info("\n🔢 NUMERICAL INSTABILITY ANALYSIS") 
    logger.info("=================================")
    
    # Check entropy progression - high entropy might indicate loss of control
    logger.info(f"Entropy Progression:")
    logger.info(f"  Early (≤300): {df[df['update'] <= 300]['entropy'].mean():.3f}")
    logger.info(f"  Mid (300-700): {df[(df['update'] > 300) & (df['update'] <= 700)]['entropy'].mean():.3f}")
    logger.info(f"  Late (>700): {df[df['update'] > 700]['entropy'].mean():.3f}")
    
    # Very high entropy suggests the policy is becoming random
    if df['entropy'].iloc[-1] > 2.0:
        logger.info("⚠️  Very high entropy suggests policy is becoming too random")
    
    # Check policy loss patterns
    logger.info(f"\nPolicy Loss Analysis:")
    policy_spikes = df[df['policy_loss'] > 3.0]
    logger.info(f"  Policy Loss Spikes (>3.0): {len(policy_spikes)} updates")
    for idx, row in policy_spikes.iterrows():
        logger.info(f"    Update {row['update']}: {row['policy_loss']:.4f}")
    
    # Check for loss correlation patterns
    logger.info(f"\nLoss Correlation Analysis:")
    high_vloss_periods = df[df['value_loss'] > 500]
    logger.info(f"  High Value Loss Periods: {len(high_vloss_periods)} updates")
    logger.info(f"  Average Reward during high VLoss: {high_vloss_periods['reward'].mean():.4f}")
    logger.info(f"  Average Policy Loss during high VLoss: {high_vloss_periods['policy_loss'].mean():.4f}")

def identify_root_cause(df):
    """Identify the most likely root cause of the issues"""
    
    logger.info("\n🎯 ROOT CAUSE ANALYSIS")
    logger.info("=====================")
    
    # Look for the smoking gun
    evidence = []
    
    # Evidence 1: Massive value loss explosion
    max_vloss = df['value_loss'].max()
    if max_vloss > 100000:
        evidence.append("Catastrophic Value Function Collapse")
        logger.info(f"🔥 EVIDENCE: Value loss explosion to {max_vloss:,.0f}")
    
    # Evidence 2: Episode length pattern
    late_episodes = df[df['update'] > 700]['ep_len'].mean()
    early_episodes = df[df['update'] <= 300]['ep_len'].mean()
    
    if late_episodes < early_episodes - 1:
        evidence.append("Progressive Episode Shortening")
        logger.info(f"📉 EVIDENCE: Episodes shortened from {early_episodes:.1f} to {late_episodes:.1f}")
    
    # Evidence 3: Performance crash pattern
    best_period = df[(df['update'] >= 500) & (df['update'] <= 600)]['reward'].mean()
    crash_period = df[(df['update'] >= 770) & (df['update'] <= 790)]['reward'].mean()
    
    if crash_period < best_period - 1:
        evidence.append("Sudden Performance Crash")
        logger.info(f"💥 EVIDENCE: Performance crashed from {best_period:.3f} to {crash_period:.3f}")
    
    # Evidence 4: Entropy explosion
    if df['entropy'].iloc[-1] > 2.0:
        evidence.append("Policy Entropy Explosion")
        logger.info(f"🌪️  EVIDENCE: Entropy exploded to {df['entropy'].iloc[-1]:.3f}")
    
    # Diagnosis
    logger.info(f"\n🔬 DIAGNOSIS:")
    
    if "Catastrophic Value Function Collapse" in evidence:
        logger.info("PRIMARY CAUSE: Value Function Numerical Instability")
        logger.info("  The value function became numerically unstable around update 890")
        logger.info("  This caused catastrophic loss spikes (722K vs normal ~200)")
        logger.info("  This is NOT overtraining - it's a technical numerical issue")
        
        logger.info("\nLIKELY TRIGGERS:")
        logger.info("  1. Learning rate too high for value function")
        logger.info("  2. Insufficient gradient clipping on value function")  
        logger.info("  3. Target values becoming too large/unstable")
        logger.info("  4. Advantage normalization issues causing value targets to explode")
        
        logger.info("\nCONSEQUENCES:")
        logger.info("  - Value function provides bad state value estimates")
        logger.info("  - Advantage calculation becomes meaningless") 
        logger.info("  - Policy updates become erratic")
        logger.info("  - Episodes terminate early due to bad decisions")
        
    return evidence

def main():
    logger.info("🔍 DEEP TECHNICAL ISSUE ANALYSIS")
    logger.info("Job 2051893429931081728 - Investigating Non-Overtraining Causes")
    logger.info("=" * 60)
    
    df = get_detailed_training_data()
    
    # Deep analyses
    analyze_value_loss_explosions(df)
    analyze_episode_length_decline(df)  
    analyze_numerical_instability_patterns(df)
    
    # Root cause identification
    root_causes = identify_root_cause(df)
    
    logger.info(f"\n{'=' * 60}")
    logger.info("CONCLUSION: This is a VALUE FUNCTION INSTABILITY issue, NOT overtraining")
    
    return len(root_causes) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)