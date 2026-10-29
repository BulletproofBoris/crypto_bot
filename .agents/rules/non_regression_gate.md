# Non-Regression Evolution Gate Rule

Whenever modifying trading execution logic, order placement rules, carousel logic, or RL model weights:
1. **Sim-to-Live Parity (`100%`):** Every rule change in `_tools/live_rl_trader.py` MUST be identically implemented in `SafeTradeL2ExecutionEnv.step` inside `_tools/train_rl_execution.py`. Never patch `live_rl_trader.py` alone.
2. **Mandatory Benchmark Gate:** Run `.venv/bin/python _tools/regression_gate.py` across the 5 Golden Regimes (`R1_PUMP_BREAKOUT`, `R2_TOXIC_DUMP`, `R3_WIDE_SPREAD_SCALP`, `R4_TIGHT_1TICK_CHOP`, `R5_UNDERWATER_RECOVERY`).
3. **Pass Requirement:** Only deploy to the VPS (`start_live_daemon`) and update `regression_baseline_passport.json` (`--save-baseline`) if the gate reports `✅ ДОПУЩЕНО К ДЕПЛОЮ (Non-Regression Gate PASSED)`.
