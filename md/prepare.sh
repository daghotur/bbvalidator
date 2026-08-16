#!/usr/bin/env bash
# Подготовка одной системы к МД: pdb2gmx -> бокс -> сольватация -> ионы -> EM -> NVT -> NPT
# Использование: md/prepare.sh <input.pdb> <workdir>
set -euo pipefail
# Путь к входному PDB разрешаем ДО смены каталога: ниже идёт cd в рабочую
# директорию, после которого относительный путь указывал бы в никуда.
SRC=$(realpath "$1"); WD=$2
MDP="$(cd "$(dirname "$0")" && pwd)/mdp"
mkdir -p "$WD"; cd "$WD"
cp "$SRC" input.pdb
LOG=prep.log; : > $LOG

# Силовое поле amber99sb-ildn (15), вода TIP3P (1). Водороды исходной структуры игнорируем.
gmx pdb2gmx -f input.pdb -o proc.gro -p topol.top -i posre.itp \
    -ff amber99sb-ildn -water tip3p -ignh >>$LOG 2>&1
# Додекаэдрический бокс, 1.0 нм от белка до границы
gmx editconf -f proc.gro -o box.gro -c -d 1.0 -bt dodecahedron >>$LOG 2>&1
gmx solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top >>$LOG 2>&1
# 0.15 М NaCl + нейтрализация заряда; ионы ставим в группу SOL (13)
gmx grompp -f "$MDP/em.mdp" -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 >>$LOG 2>&1
echo SOL | gmx genion -s ions.tpr -o ions.gro -p topol.top \
    -pname NA -nname CL -neutral -conc 0.15 >>$LOG 2>&1
# Минимизация
gmx grompp -f "$MDP/em.mdp" -c ions.gro -p topol.top -o em.tpr -maxwarn 2 >>$LOG 2>&1
gmx mdrun -deffnm em -ntmpi 1 -ntomp "${OMP:-10}" -nb gpu >>$LOG 2>&1
# NVT
gmx grompp -f "$MDP/nvt.mdp" -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2 >>$LOG 2>&1
gmx mdrun -deffnm nvt -ntmpi 1 -ntomp "${OMP:-10}" -nb gpu -pme gpu -bonded gpu >>$LOG 2>&1
# NPT
gmx grompp -f "$MDP/npt.mdp" -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2 >>$LOG 2>&1
gmx mdrun -deffnm npt -ntmpi 1 -ntomp "${OMP:-10}" -nb gpu -pme gpu -bonded gpu >>$LOG 2>&1
ATOMS=$(awk 'NR==2{print $1}' npt.gro)
echo "PREP OK  atoms=$ATOMS  $WD"
