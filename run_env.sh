#!/bin/bash
set -euo pipefail

AGENTS_DIR="agents"

ALLOW_ENVS=(
  # MountainCar-v0
  Catch-bsuite
  DeepSea-bsuite
  FourRooms-misc
  Pendulum-v1
  Freeway-MinAtar
)

# 8 seeds（PPO 和 DHVL 都用）
seeds=(111 222 333 444 555 666 777 888)

# 固定 tau（DHVL 用；如果 PPO 也要用 tau，就一起传）
taus=(0.6)

# PPO 要扫的 value_update_iter
ppo_value_update_iters=(4 8 16 32)

# DHVL 只跑一次 value_update_iter=4
dhvl_value_update_iter=4

extra_fixed_args=()

for env_name in "${ALLOW_ENVS[@]}"; do
  env_dir="${AGENTS_DIR}/${env_name}"
  cfg="${env_dir}/ppo.yaml"

  if [ ! -d "$env_dir" ]; then
    echo "[SKIP] ${env_name}: directory not found"
    continue
  fi

  if [ ! -f "$cfg" ]; then
    echo "[SKIP] ${env_name}: no ppo.yaml"
    continue
  fi

  echo "============================================================"
  echo "[ENV] ${env_name}"
  echo "============================================================"

  # 1) PPO: 扫 value_update_iter=(1 4 8 16)
  for vui in "${ppo_value_update_iters[@]}"; do
    echo "------------------------------"
    echo "[PPO] value_update_iter=${vui}"
    echo "------------------------------"

    for seed in "${seeds[@]}"; do
      python -m train \
        -config "${cfg}" \
        --seed_id "${seed}" \
        --algo "ppo" \
        --value_update_iter "${vui}" \
        --actor_update_iter 1 \
        "${extra_fixed_args[@]}" \
        "$@"
    done
  done

  # 2) DHVL: tau 扫（当前只有 0.6），value_update_iter 只跑 4
  for tau in "${taus[@]}"; do
    echo "------------------------------"
    echo "[DHVL] tau=${tau} value_update_iter=${dhvl_value_update_iter}"
    echo "------------------------------"

    for seed in "${seeds[@]}"; do
      python -m train \
        -config "${cfg}" \
        --seed_id "${seed}" \
        --algo "dhvl" \
        --tau "${tau}" \
        --value_update_iter "${dhvl_value_update_iter}" \
        --actor_update_iter 1 \
        "${extra_fixed_args[@]}" \
        "$@"
    done
  done
done
