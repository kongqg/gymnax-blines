#!/bin/bash
set -euo pipefail

AGENTS_DIR="agents"

# 8 seeds（PPO 和 DHVL 都用）
seeds=(111 222 333 444 555 666 777 888)

# DHVL tau grid: 0.1..0.9
taus=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# 固定额外参数（可选）
extra_fixed_args=()

# 遍历 agents 下所有环境目录
for env_dir in "$AGENTS_DIR"/*/; do
  # 防止没有匹配时传入字面量
  [ -d "$env_dir" ] || continue

  env_name="$(basename "$env_dir")"
  cfg="${env_dir%/}/ppo.yaml"

  # 如果没有 ppo.yaml，跳过
  if [ ! -f "$cfg" ]; then
    echo "[SKIP] ${env_name}: no ppo.yaml at ${cfg}"
    continue
  fi

  echo "============================================================"
  echo "[ENV] ${env_name}  cfg=${cfg}"
  echo "============================================================"

  # 1) PPO：8 seeds
  for seed in "${seeds[@]}"; do
    echo "------------------------------"
    echo "[RUN] algo=ppo env=${env_name} seed=${seed} cfg=${cfg}"
    echo "------------------------------"
    python -m train \
      -config "${cfg}" \
      --seed_id "${seed}" \
      --algo "ppo" \
      "${extra_fixed_args[@]}" \
      "$@"
  done

  # 2) DHVL：8 seeds × tau grid
  for tau in "${taus[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "------------------------------"
      echo "[RUN] algo=dhvl env=${env_name} tau=${tau} seed=${seed} cfg=${cfg}"
      echo "------------------------------"
      python -m train \
        -config "${cfg}" \
        --seed_id "${seed}" \
        --algo "dhvl" \
        --tau "${tau}" \
        "${extra_fixed_args[@]}" \
        "$@"
    done
  done
done
