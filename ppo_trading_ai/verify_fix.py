#!/usr/bin/env python3
"""
Verify Senior Dev Fix - Backtest Sanity Check
Test the fixed environment with known actions to verify cost calculations
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from ppo_env_fixed import PPOTradingEnvironment, TradingConfig

def verify_cost_calculations():
    """Verify that costs are reasonable and not catastrophic"""
    print("🔍 VERIFYING SENIOR DEV FIX - Cost Calculation Sanity Check")
    print("=" * 60)
    
    # Create environment with known configuration
    config = TradingConfig(
        initial_balance=10000.0,
        max_leverage=3.0,
        taker_fee=0.0004,  # 4 bps
        slippage_bp=1.0    # 1 bp
    )
    
    env = PPOTradingEnvironment(config)
    obs, info = env.reset(seed=42)
    
    print("Initial state:")
    print(f"  Equity: ${info['equity']:,.2f}")
    print(f"  Position N: ${info['N']:,.2f}")
    print(f"  Leverage: {info['leverage']:.2f}x")
    print()
    
    # Test sequence: 0 → 1x → 3x → -2x → 0
    test_actions = [0.33, 1.0, -0.67, -0.67, 0.0]  # Target leverage ratios
    total_costs = 0.0
    
    for i, action in enumerate(test_actions):
        old_equity = env.equity
        old_N = env.N
        
        obs, reward, terminated, done, info = env.step(np.array([action]))
        
        cost = info['cost']
        new_equity = info['equity']
        new_N = info['N']
        leverage = info['leverage']
        turnover = info['turnover']
        
        cost_ratio = cost / max(old_equity, 1.0)
        
        print(f"Step {i+1}: Action {action:+.2f}")
        print(f"  Position: ${old_N:,.0f} → ${new_N:,.0f} (turnover: ${turnover:,.0f})")
        print(f"  Leverage: {abs(old_N)/max(old_equity,1):.2f}x → {leverage:.2f}x")
        print(f"  Cost: ${cost:.2f} ({cost_ratio:.4%} of equity)")
        print(f"  Equity: ${old_equity:,.2f} → ${new_equity:,.2f}")
        print(f"  Reward: {reward:.4f}")
        
        total_costs += cost
        
        # Sanity checks
        if cost_ratio > 0.05:  # 5% cost per trade is excessive
            print(f"  ❌ WARNING: Cost ratio {cost_ratio:.2%} seems high")
        else:
            print(f"  ✅ Cost ratio {cost_ratio:.4%} is reasonable")
        
        if abs(leverage) > config.max_leverage * 1.01:
            print(f"  ❌ WARNING: Leverage {leverage:.2f}x exceeds max {config.max_leverage}x")
        else:
            print(f"  ✅ Leverage {leverage:.2f}x within limits")
        
        print()
        
        if terminated:
            print("❌ Environment terminated (bankruptcy!)")
            break
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Total costs: ${total_costs:.2f}")
    print(f"Final equity: ${env.equity:,.2f}")
    print(f"Net P&L: ${env.equity - config.initial_balance:,.2f}")
    print(f"Cost as % of initial capital: {total_costs/config.initial_balance:.2%}")
    print()
    
    # Senior dev expectation: costs should be reasonable
    if total_costs > config.initial_balance * 0.10:  # 10% of initial capital
        print("❌ FAILED: Total costs exceed 10% of initial capital - still broken!")
        return False
    else:
        print("✅ PASSED: Total costs are reasonable")
        return True

def verify_leverage_action_mapping():
    """Verify that actions correctly map to leverage targets"""
    print("\n🎯 VERIFYING ACTION → LEVERAGE MAPPING")
    print("=" * 60)
    
    config = TradingConfig(initial_balance=10000.0, max_leverage=3.0)
    env = PPOTradingEnvironment(config)
    obs, info = env.reset(seed=42)
    
    test_cases = [
        (-1.0, -3.0, "Max short"),
        (-0.5, -1.5, "Half short"), 
        (0.0, 0.0, "Neutral"),
        (0.5, 1.5, "Half long"),
        (1.0, 3.0, "Max long")
    ]
    
    all_passed = True
    
    for action, expected_leverage, description in test_cases:
        # Reset to clean state
        obs, info = env.reset(seed=42)
        
        # Take action
        obs, reward, terminated, done, info = env.step(np.array([action]))
        
        actual_leverage = info['leverage'] * np.sign(info['N'])  # Include sign
        
        print(f"{description:12}: action={action:+.1f} → L={actual_leverage:+.1f}x (expected: {expected_leverage:+.1f}x)")
        
        if abs(actual_leverage - expected_leverage) < 0.1:
            print("            ✅ PASS")
        else:
            print(f"            ❌ FAIL: Expected {expected_leverage:+.1f}x, got {actual_leverage:+.1f}x")
            all_passed = False
    
    print()
    if all_passed:
        print("✅ PASSED: All action → leverage mappings correct")
        return True
    else:
        print("❌ FAILED: Action → leverage mapping broken")
        return False

def main():
    """Run all verification tests"""
    print("🧪 SENIOR DEV FIX VERIFICATION")
    print("Testing USD notional position system vs. broken leverage system")
    print()
    
    test1_passed = verify_cost_calculations()
    test2_passed = verify_leverage_action_mapping()
    
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION RESULTS:")
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED - Senior dev fix is working correctly!")
        print("✅ Ready for training with reasonable cost structure")
        return True
    else:
        print("❌ SOME TESTS FAILED - Do not proceed to training!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)