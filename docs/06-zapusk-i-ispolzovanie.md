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
python -m training.hybrid                          # гибрид (d_model=192)
python baselines/train_baseline.py --encoder mlp
python baselines/train_baseline.py --encoder gps
```

Гиперпараметры и протокол — в модуле [03](03-model-i-obuchenie.md). Чекпоинты: `checkpoints/best_model.pth` (по составной val-метрике) и `checkpoints/last_model.pth` (для возобновления); базлайны — `checkpoints/baseline_{mlp,gps}_{best,last}.pth`. Логи: `runs/…` (`tensorboard --logdir runs/`).

## Оценка

**Полная батарея на сплите** (test по умолчанию):

```bash
python -m evaluation.eval_model -c checkpoints/best_model.pth --split test -o results/eval_results_hybrid.json
```

Метрики: Accuracy, ROC-AUC, PR-AUC, ECE, precision/recall; per-strategy (specificity + AUC «нативы против семейства»); confusion 6×6 failure_mode; MSE auxiliary-голов. Архитектура чекпоинта (гибрид/MLP/GPS) определяется автоматически по ключам state_dict.

**Скоринг выходов внешних генераторов** (OOD):

```bash
python -m evaluation.eval_generated \
    --dirs "RFdiffusion=data/ood/rfdiffusion" \
    --pattern "**/*.pdb" \
    -o results/eval_results_generated.json
```

Для каждой структуры: P(fold) + MC-Dropout неопределённость; плюс референсные группы native/decoy из test. Скрипт знает стандартные MotifBench-директории в `data/ood/` и запускается без `--dirs`.

**Корреляция с ground-truth качеством** (MotifBench self-consistency):

```bash
python -m analysis.scrmsd        # Spearman/Pearson P(fold) ↔ scRMSD по генераторам + графики
```

**Метка проекта.** scRMSD скаффолда = $\min_k$ по восьми последовательностям (спецификация MotifBench, docs/07). Агрегатор задан один раз константой `SCRMSD_AGG` в `common/motifbench.py`; там же лежит единственный парсер `*_eval_results.csv`, которым пользуются и анализы, и обучающие скрипты, поэтому определение метки не может разъехаться между ними.

**Анализы поверх меток** (все пишут JSON + CSV, обучения не требуют):

```bash
python -m analysis.label_choice   # выбор агрегатора: mean против min, цена перехода
python -m analysis.oracle_ceiling # потолок оракула, split-half по последовательностям
python -m analysis.relabel        # ранжирование моделей против потолка
python -m analysis.motif_bias     # внутримотивный lift, исключение насыщенных мотивов
python -m analysis.second_oracle  # перенос на AF2 как независимый рефолдер
python -m analysis.baselines      # дешёвые бейзлайны против модели
python -m analysis.economics      # экономия дорогих проверок по сложности мишени
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

Чекпоинты, обученные до пересборки 2026-08, несовместимы с текущей архитектурой (убрана hbond-голова, добавлен PCA-буфер) — `inference.py` сообщит об этом явно. Чекпоинты до расширения парного канала (20 → 29 признаков) загружаются: `expand_pair_embed` дополняет вход проекции нулями, и модель считает ровно то же, что и раньше.

## Фильтрация по дизайнуемости

`filter_designability.py` батчево скорит каталог структур и для каждой выдаёт вердикт PASS/FAIL со списком заваленных гейтов. Две модели с разделёнными ролями (раздел 5.4): ранжирование и вердикт — `soft_model.pth` (мягкая метка log1p(scRMSD), детерминированный скор, без сигмоиды и MC-усреднения); диагностика похожести на натив — `soft_model_mt.pth` (мультизадачное дообучение с native-якорем: калиброванный scRMSD, RMSD aux-головы, стерика, failure mode), диагностические колонки в вердикте не участвуют, расхождение моделей помечается маркером:

```bash
python filter_designability.py -i data/ood/rfdiffusion/scaffolds --max-scrmsd 2.0 -o results/rfdiff_filter.csv
```

Обязательный гейт — `--max-scrmsd` в шкале ранжирующей модели. Пороги пересчитаны после перехода на метку $\min_k$ (docs/05, 5.6) и сместились вниз; на потоке RFdiffusion:

| порог, Å | покрытие | точность |
|---:|---:|---:|
| 1.0 | 27.1% | 0.966 |
| 1.5 | 54.1% | 0.896 |
| 2.0 | 73.6% | 0.830 |
| 3.0 | 93.2% | 0.745 |

То есть 1.0 Å — консервативный режим, 2.0 Å — режим покрытия. **Пороги привязаны к конкретному чекпоинту**: шкала предсказаний сдвигается при каждом переобучении (в прогоне 13.08 те же 2.0 Å давали 44% покрытия), поэтому переносить значения между версиями весов нельзя — таблицу нужно перестраивать прогоном фильтра на потоке. Опциональные: `--max-clashes` (сырые конфликты геометрии Cβ < 3.5 Å, |i-j| ≥ 3), `--max-uncertainty` (MC-дисперсия ранжирующей модели, только при `--mc-runs > 1`). Независимо от гейтов verbose-вывод и CSV содержат все метрики: pred scRMSD, MC-дисперсию, диагностику похожести на натив, вероятность стерики, RMSD aux-головы, failure mode, сырые счётчики конфликтов и H-связей.

## Интерактивные тетрадки

```bash
uv run jupyter lab     # jupyterlab ставится dev-зависимостью при uv sync
```

- `notebooks/01-kak-rabotaet.ipynb` — проход пайплайна по шагам на живых данных: контракт фронтенда (31 узловой + 29 парных признаков), разбор парного признака на блок расстояний и блок ориентации, **проверка хиральности** (отражённый остов: блок расстояний совпадает побитово, ориентация — нет, скор рушится), энкодер и головы, по-остаточный профиль lDDT, скоринг всех образцов из `data/` с объяснением слепой зоны на нативах; финальный раздел — атрибуция признаков (Integrated Gradients, `explain_protein()`): для любого PDB показывает, какие из 31 узловых признаков и какие остатки сильнее всего ухудшают предсказанный scRMSD;
- `notebooks/02-metriki-i-primery.ipynb` — сводка по артефактам анализов в порядке логики вопросов: выбор метки (mean против min), потолок оракула и доля забранного сигнала, дешёвые бейзлайны, перенос на AF2, экономика скрининга по сложности мишени; ничего не обучает и не скорит, только читает `analysis_*.json`;
- `notebooks/03-benchmark.ipynb` — разбивка латентности по компонентам (фронтенд / граф / трансформер / головы), батчевый режим, цена MC-Dropout и замер того, что скрининг упирается в парсинг PDB, а не в модель.

Все три исполняются целиком и хранятся с выводами; для пересборки достаточно `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.

## Бенчмарк скорости

```bash
python -m evaluation.benchmark -i data/3MYC.pdb -c checkpoints/best_model.pth -m 16 -n 50
```

5 прогревочных + `n` замеров, `m` MC-проходов; физический фронтенд считается один раз на структуру.
