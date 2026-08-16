#!/usr/bin/env bash
# Продуктивные прогоны по всем подготовленным системам, последовательно.
# Использование: md/produce.sh <ns>
set -uo pipefail
NS=${1:-10}
STEPS=$(python3 -c "print(int($NS/0.002*1000))")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MDP="$ROOT/md/mdp/md.mdp"
echo "=== продукция $NS нс ($STEPS шагов) на систему ==="
while IFS=$'\t' read -r TAG SRC; do
  WD="$ROOT/md/runs/$TAG"
  [ -f "$WD/npt.gro" ] || { echo "SKIP $TAG (нет npt.gro)"; continue; }
  [ -f "$WD/md.gro" ] && { echo "DONE $TAG (уже посчитано)"; continue; }
  cd "$WD"
  sed "s/^nsteps.*/nsteps = $STEPS/" "$MDP" > md_run.mdp
  gmx grompp -f md_run.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 2 >>prep.log 2>&1
  T0=$(date +%s)
  gmx mdrun -deffnm md -ntmpi 1 -ntomp 10 -nb gpu -pme gpu -bonded gpu -update gpu >>md.log 2>&1
  RC=$?; T1=$(date +%s)
  PERF=$(grep -A1 'ns/day' md.log | tail -1 | awk '{print $2}')
  if [ $RC -eq 0 ] && [ -f md.gro ]; then
    printf "OK   %-28s %4s мин  %s нс/день\n" "$TAG" "$(( (T1-T0)/60 ))" "$PERF"
  else
    printf "FAIL %-28s rc=%s\n" "$TAG" "$RC"
  fi
done < "$ROOT/md/systems.tsv"
echo "=== продукция завершена ==="
