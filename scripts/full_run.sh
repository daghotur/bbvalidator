#!/usr/bin/env bash
# Полный прогон на пересобранном сплите: обучение всех стадий, затем метрики.
# Каждый шаг пишет свой лог; при падении шага прогон останавливается.
#
#   bash scripts/full_run.sh [каталог_логов]
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGS="${1:-logs/full_run}"
mkdir -p "$LOGS" results

step() {
  local name="$1"; shift
  local log="$LOGS/$name.log"
  if [ -f "$LOGS/$name.done" ]; then
    echo "[$(date +%H:%M:%S)] пропуск $name (уже сделан)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] старт $name: $*"
  local start=$SECONDS
  if "$@" > "$log" 2>&1; then
    touch "$LOGS/$name.done"
    echo "[$(date +%H:%M:%S)] готово $name за $(( (SECONDS - start) / 60 )) мин"
  else
    echo "[$(date +%H:%M:%S)] ОШИБКА $name — см. $log"; tail -20 "$log"; exit 1
  fi
}

echo "=== 1. Обучение с нуля на новом сплите ==="
step train_hybrid   $PY -u -m training.hybrid
step train_mlp      $PY -u baselines/train_baseline.py --encoder mlp
step train_gps      $PY -u baselines/train_baseline.py --encoder gps

echo "=== 2. Дообучение на метки оракула ==="
step train_soft      $PY -u -m training.soft
step train_soft_mt   $PY -u -m training.soft_multitask
step train_perres    $PY -u -m training.perresidue
step train_joint     $PY -u -m training.joint

echo "=== 3. Батарея на test ==="
step eval_hybrid $PY -u -m evaluation.eval_model -c checkpoints/best_model.pth \
      --split test -o results/eval_results_hybrid.json
step eval_mlp    $PY -u -m evaluation.eval_model -c checkpoints/baseline_mlp_best.pth \
      --split test -o results/eval_results_mlp.json
step eval_gps    $PY -u -m evaluation.eval_model -c checkpoints/baseline_gps_best.pth \
      --split test -o results/eval_results_gps.json

echo "=== 4. OOD-скоринг генераторов ==="
step eval_generated $PY -u -m evaluation.eval_generated -o results/eval_results_generated.json

echo "=== 5. Анализы ==="
step an_scrmsd        $PY -u -m analysis.scrmsd
step an_logits        $PY -u -m analysis.logits_ranking
step an_enrichment    $PY -u -m analysis.enrichment
step an_motif_bias    $PY -u -m analysis.motif_bias
step an_label_choice  $PY -u -m analysis.label_choice
step an_oracle        $PY -u -m analysis.oracle_ceiling
step an_relabel       $PY -u -m analysis.relabel
step an_perresidue    $PY -u -m analysis.perresidue
step an_second_oracle $PY -u -m analysis.second_oracle
step an_baselines     $PY -u -m analysis.baselines
step an_economics     $PY -u -m analysis.economics

echo "[$(date +%H:%M:%S)] ВСЁ ГОТОВО"
