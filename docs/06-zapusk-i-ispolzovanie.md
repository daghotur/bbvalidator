# 06 — Запуск и использование

## Установка

```bash
uv sync          # Python >= 3.13, зависимости из pyproject.toml
pytest           # 12 тестов фронтенда (CPU, < 1 с)
```

или вручную: `pip install torch torch-geometric biotite scipy pandas h5py tensorboard requests tqdm`.

## Пайплайн данных (с нуля)

Все команды из корня репозитория; данные кладутся в `dataset/`.

```bash
cd dataset
python build_positive_dataset.py   # 1. RCSB: X-Ray ≤ 2.0 Å, L 50–700 → positive_proteins.h5
python build_negative_dataset.py   # 2. 2 декоя на цепь, 7 стратегий, seed=42 → negative_proteins.h5
python compute_targets.py          # 3. steric_target на GPU (идемпотентно)
python make_split.py               # 4. манифесты + сплиты по group_id
cd ..
python -m preprocess.fit_pca       # 5. PCA фронтенда → dataset/pca_components.pth
```

`pca_components.pth` не коммитится (воспроизводится пятым шагом) и обязателен для обучения и инференса.

## Обучение

```bash
python train_model.py                          # гибрид (d_model=192)
python baselines/train_baseline.py --encoder mlp
python baselines/train_baseline.py --encoder gps
```

Гиперпараметры и протокол — в модуле [03](03-model-i-obuchenie.md). Чекпоинты: `checkpoints/best_model.pth` (по составной val-метрике) и `checkpoints/last_model.pth` (для возобновления); базлайны — `checkpoints/baseline_{mlp,gps}_{best,last}.pth`. Логи: `runs/…` (`tensorboard --logdir runs/`).

## Оценка

**Полная батарея на сплите** (test по умолчанию):

```bash
python eval_model.py -c checkpoints/best_model.pth --split test -o eval_results_hybrid.json
```

Метрики: Accuracy, ROC-AUC, PR-AUC, ECE, precision/recall; per-strategy (specificity + AUC «нативы против семейства»); confusion 6×6 failure_mode; MSE auxiliary-голов. Архитектура чекпоинта (гибрид/MLP/GPS) определяется автоматически по ключам state_dict.

**Скоринг выходов внешних генераторов** (OOD):

```bash
python eval_generated.py \
    --dirs "RFdiffusion=data/ood/rfdiffusion" \
    --pattern "**/*.pdb" \
    -o eval_results_generated.json
```

Для каждой структуры: P(fold) + MC-Dropout неопределённость; плюс референсные группы native/decoy из test. Скрипт знает стандартные MotifBench-директории в `data/ood/` и запускается без `--dirs`.

**Корреляция с ground-truth качеством** (MotifBench self-consistency):

```bash
python analysis_scrmsd.py        # Spearman/PEARSON P(fold) ↔ scRMSD по генераторам + графики
```

## Инференс отдельных PDB

```bash
python inference.py -i structure.pdb -c checkpoints/best_model.pth
python inference.py -i path/to/pdb_dir/ -c checkpoints/best_model.pth   # каталог
python inference.py -i structure.pdb --cpu -m 32                        # CPU, 32 MC-прохода
```

| Колонка | Описание |
|---|---|
| `P(Fold)` | средняя вероятность свертываемости (0–1) |
| `Uncert.` | дисперсия MC-проходов (эпистемическая неопределённость) |
| `Pred RMSD` | предсказанный RMSD до нативного аналога (Å) |
| `Len` | длина цепи |

Визуальный индикатор: ✅ `P > 0.8` · ⚠️ `P > 0.4` · ❌ `P ≤ 0.4`.

Чекпоинты, обученные до пересборки 2026-08, несовместимы с текущей архитектурой (убрана hbond-голова, добавлен PCA-буфер) — `inference.py` сообщит об этом явно.

## Бенчмарк скорости

```bash
python benchmark.py -i data/3MYC.pdb -c checkpoints/best_model.pth -m 16 -n 50
```

5 прогревочных + `n` замеров, `m` MC-проходов; физический фронтенд считается один раз на структуру.
