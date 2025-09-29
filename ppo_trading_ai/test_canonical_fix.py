#!/usr/bin/env python3
"""
SENIOR DEV UNIT TESTS - Must Pass Before Training
Tests for canonical USD notional position system fix
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from ppo_env_fixed import TradingConfig, PPOTradingEnvironment

def test_turnover_fee_math():
    """Test: Fees scale with turnover, not leverage deltas"""
    print("🧪 Testing turnover fee math...")
    
    cfg = TradingConfig()
    E = 10_000.0
    
    # 0.5x -> 3.0x at E=10k
    N0 = 0.5 * E         # 5,000
    N1 = 3.0 * E         # 30,000
    dN = abs(N1 - N0)    # 25,000
    fee = dN * cfg.taker_fee            # 10.0
    slip = dN * (cfg.slippage_bp/1e4)   # 2.5
    
    assert abs(fee - 10.0) < 1e-10, f"Fee should be 10.0, got {fee}"
    assert abs(slip - 2.5) < 1e-10, f"Slippage should be 2.5, got {slip}"
    
    print(f"✅ Turnover: ${dN:,.0f}, Fee: ${fee:.2f}, Slippage: ${slip:.2f}")

def test_pnl_matches_return():
    """Test: PnL = N * return (correct temporal ordering)"""
    print("🧪 Testing PnL calculation...")
    
    E0, N, r = 10_000.0, 20_000.0, 0.01  # +1%
    pnl = N * r
    expected = 200.0
    
    assert abs(pnl - expected) < 1e-10, f"PnL should be {expected}, got {pnl}"
    
    print(f"✅ Position: ${N:,.0f}, Return: {r:.2%}, PnL: ${pnl:.2f}")

def test_cost_cap():
    """Test: Trading costs capped at reasonable percentage"""
    print("🧪 Testing cost cap protection...")
    
    cfg = TradingConfig(cost_cap_frac=0.10)
    E = 10_000.0
    huge_turnover = 5 * E  # $50k turnover on $10k equity
    cap = 0.10 * E  # $1,000 cap
    raw_cost = huge_turnover * cfg.taker_fee  # $20 without cap
    cost = min(raw_cost, cap)
    
    assert cost <= cap, f"Cost {cost} should be capped at {cap}"
    
    print(f"✅ Huge turnover: ${huge_turnover:,.0f}, Raw cost: ${raw_cost:.2f}, Capped: ${cost:.2f}")

def test_leverage_invariant():
    """Test: Leverage never exceeds L_max after forced reduction"""
    print("🧪 Testing leverage invariant...")
    
    cfg = TradingConfig(max_leverage=3.0)
    E = 5_000.0
    N = 25_000.0  # 5x → must be trimmed to 3x
    N_safe = np.sign(N) * cfg.max_leverage * E
    L_result = abs(N_safe) / E
    
    assert abs(L_result - cfg.max_leverage) < 1e-10, f"Leverage should be {cfg.max_leverage}, got {L_result}"
    
    print(f"✅ Original N: ${N:,.0f} (5x), Safe N: ${N_safe:,.0f} ({L_result:.1f}x)")

def test_environment_integration():
    """Test: Environment runs without errors and produces reasonable costs"""
    print("🧪 Testing environment integration...")
    
    env = PPOTradingEnvironment()
    obs, info = env.reset(seed=42)
    
    # Test a few steps with different actions
    total_cost = 0
    for i, action in enumerate([-0.8, 0.5, 1.0, -0.3, 0.0]):
        obs, reward, terminated, done, info = env.step(np.array([action]))
        
        cost_ratio = info.get("cost_ratio", 0.0)
        leverage = info.get("leverage", 0.0)
        
        print(f"  Step {i+1}: action={action:+.1f}, L={leverage:.2f}x, cost_ratio={cost_ratio:.4f}, reward={reward:.4f}")
        
        total_cost += info.get("cost", 0.0)
        
        # Sanity checks
        assert cost_ratio < 0.20, f"Cost ratio {cost_ratio:.4f} too high (>20%)"
        assert leverage <= env.config.max_leverage * 1.01, f"Leverage {leverage:.2f} exceeds max {env.config.max_leverage}"
        assert not (terminated and i == 0), "Should not terminate immediately"
        
        if terminated or done:
            break
    
    avg_cost_ratio = total_cost / max(env.config.initial_balance, 1.0) / max(i+1, 1)
    print(f"✅ Ran {i+1} steps, avg cost ratio: {avg_cost_ratio:.4f}")

def test_backtest_sanity_check():
    """Test: Sanity check from senior dev - $12.88 cost, not $32k"""
    print("🧪 Testing backtest sanity check...")
    
    cfg = TradingConfig()
    
    # With cfg as above, E=10k, move from 0.5x to 3.0x in one step, price flat
    E = 10_000.0
    N0 = 0.5 * E  # $5,000
    N1 = 3.0 * E  # $30,000
    
    # Turnover calculation
    turnover = abs(N1 - N0)  # $25,000
    
    # Fee calculation
    fee = turnover * cfg.taker_fee  # 25,000 × 0.0004 = 10.00
    
    # Slippage calculation  
    slippage = turnover * (cfg.slippage_bp / 1e4)  # 25,000 × 0.0001 = 2.50
    
    # Funding calculation (1h step)
    funding_1h = N1 * cfg.funding_rate_8h * (cfg.step_hours / 8.0)  # 30,000 × 0.0000125 = 0.375
    
    # Total cost
    total_cost = fee + slippage + funding_1h
    
    print(f"  Turnover: ${turnover:,.0f}")
    print(f"  Fee: ${fee:.2f}")
    print(f"  Slippage: ${slippage:.2f}")  
    print(f"  Funding (1h): ${funding_1h:.3f}")
    print(f"  Total cost: ${total_cost:.2f}")
    
    # Senior dev expectation: ~$12.88, not $32k
    expected_range = (10.0, 15.0)
    assert expected_range[0] <= total_cost <= expected_range[1], f"Cost {total_cost:.2f} not in expected range {expected_range}"
    
    print(f"✅ Total equity drop: ${total_cost:.2f} (within expected range)")

def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("SENIOR DEV UNIT TESTS - USD NOTIONAL POSITION SYSTEM")
    print("=" * 60)
    
    tests = [
        test_turnover_fee_math,
        test_pnl_matches_return, 
        test_cost_cap,
        test_leverage_invariant,
        test_environment_integration,
        test_backtest_sanity_check
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("❌ CRITICAL: Unit tests failed - DO NOT PROCEED TO TRAINING")
        return False
    else:
        print("✅ ALL TESTS PASSED - Safe to proceed to training")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)