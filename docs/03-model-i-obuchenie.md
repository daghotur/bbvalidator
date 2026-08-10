# 03 — Модель и обучение

## 3.1 Архитектура

```mermaid
flowchart TD
    INPUT["coords [B, L, 3, 3] + mask"]
    subgraph FRONTEND["BiophysicalFrontend · no_grad · заморожен"]
        GEO["BackboneGeometry<br/>торсионы · Рамачандран · клэши · H-связи<br/>→ 10 признаков"]
        FOLD["DesignabilityProxies<br/>упаковка · экспонированность · PCA-frag<br/>→ 21 признак"]
        PAIR["PairFeatureBuilder<br/>RBF-16 · kNN-16 · seq-sep<br/>→ 20 признаков"]
    end
    NODE["node_feats [B, L, 31]"]
    PAIRF["kNN-рёбра: edge_indices + edge_attrs [E, 20]"]
    subgraph ENCODER["HybridProteinEncoder"]
        MPNN["PyGGraphMessageLayer × 2<br/>GRU-gate MPNN + gradient checkpointing"]
        TRANS["TransformerEncoder × 4<br/>Pre-LN · d_model = 192 · 8 голов"]
    end
    POOL["MultiHeadAttentionPooling · 4 головы<br/>softmax по L → [B, 192]"]
    subgraph HEADS["ProteinMultiTaskHeads"]
        H1["fold_logit"]
        H2["rmsd"]
        H3["steric"]
        H4["failure_mode (6 классов)"]
    end
    LOSS["DynamicMultiTaskLoss<br/>Kendall & Gal, 4 задачи"]

    INPUT --> FRONTEND
    GEO --> NODE
    FOLD --> NODE
    PAIR --> PAIRF
    NODE --> MPNN
    PAIRF --> MPNN
    MPNN --> TRANS
    TRANS --> POOL
    POOL --> HEADS
    HEADS --> LOSS
```

## 3.2 Гибридный энкодер

Идея гибридизации: локальные контакты и глобальный контекст обрабатываются специализированными механизмами.

**MPNN на kNN-графе (2 слоя).** Каждый остаток обменивается сообщениями с 16 пространственными соседями. Сообщение — функция признаков узла-источника и парных признаков ребра (20-мерных); обновление узла через GRU-гейт (сигналы «забыть/обновить» контролируют, какую долю старой информации сохранить). Упаковка графа: паддинг удаляется, графы батча стыкуются в один через офсеты индексов — вычисления идут только по валидным узлам. Для экономии памяти MPNN-слои обёрнуты в **gradient checkpointing** (`use_reentrant=False`).

**Transformer Encoder (4 слоя, Pre-LN).** Обрабатывает последовательность целиком и ловит дальнодействующие корреляции, недоступные локальному графу. `src_key_padding_mask` исключает паддинг; обучение в bf16 (см. 3.6).

## 3.3 Пулинг и головы

**MultiHeadAttentionPooling (4 головы):** обучаемые запрос-векторы attended-пулинга собирают взвешенную сумму по остаткам (softmax с маскированием паддинга $-\!10^9$), затем головы конкатенируются и проецируются в $d_{model}$. Даёт представление всей структуры $[B, d_{model}]$.

**ProteinMultiTaskHeads:** четыре независимых MLP-головы (fold → 1 логит, rmsd → 1, steric → 1, failure_mode → 6). Класс 5 (`unknown_negative`) в данных отсутствует — выход зарезервирован под будущие OOD-негативы.

## 3.4 Функция потерь: гомоскедастическая неопределённость

Четыре задачи разной природы (классификация + регрессии в разных шкалах) нельзя суммировать с ручными весами. Используется подход Kendall et al. (2018): каждая задача получает обучаемый log-варианс $s_t = \ln \sigma_t^2$:

$$\mathcal{L}_{total} = \sum_{t=1}^{4} \left( \tfrac{1}{2} e^{-s_t} \mathcal{L}_t + \tfrac{1}{2} s_t \right)$$

Второе слагаемое штрафует завышение дисперсии и не даёт лоссу уйти в $-\infty$. Задачи:

| Лосс | Формула |
|---|---|
| fold | $\operatorname{BCEWithLogits}(\hat{y}_{fold}, y_{label})$ |
| rmsd | $\operatorname{MSE}(\hat{y}_{rmsd}, \ln(1 + y_{rmsd}))$ — log1p стабилизирует хвост разрушенных структур |
| steric | $\operatorname{MSE}(\hat{y}_{steric}, y_{steric})$ |
| failure_mode | $\operatorname{CrossEntropy}(\hat{y}_{fm}, y_{fm}, \text{weight}=W_{classes})$ |

Для failure_mode передаются **веса классов по обратной частоте** train-сплита (компенсация дисбаланса; класс 5 имеет нулевой вес, т.к. в данных отсутствует). Параметры $s_t$ оптимизируются отдельной param-group с LR $10^{-3}$.

## 3.5 MC-Dropout инференс

Эпистемическая неопределённость оценивается многократным прогоном с включённым дропаутом (Gal & Ghahramani, 2016):

1. фронтенд вычисляет физические признаки **строго один раз** (`no_grad`);
2. у всех `nn.Dropout`-модулей энкодера и голов временно включается режим `train`;
3. $M$ проходов (по умолчанию 8): собирается $\{\sigma(\text{logit}_m)\}$;
4. выход: $\bar{p} = \operatorname{mean}$, $U = \operatorname{var}$;
5. флаги модулей восстанавливаются.

Поскольку каждый MC-проход гоняет только энкодер (физика уже посчитана), инференс остаётся дешёвым — см. бенчмарк в модуле 06.

## 3.6 Протокол обучения

| Параметр | Значение | Комментарий |
|---|---|---|
| $d_{model}$ | 192 | у базлайнов 128 |
| MPNN-слои / Transformer-слои | 2 / 4 | Pre-LN, 8 голов |
| Dropout | 0.15 | энкодер и головы |
| Batch size | 64 | **бакеты по длине** (шаг 50) — минимум паддинг-отходов при $L \in [50, 700]$ |
| Эпохи | 15 | cosine annealing |
| LR (модель) | $6 \times 10^{-4}$ | AdamW, weight decay $10^{-4}$; sqrt-скейлинг от базового $3 \times 10^{-4}$ при переходе batch 16 → 64 |
| LR (параметры лосса $s_t$) | $10^{-3}$ | отдельная param-group |
| Сид | 42 | torch/numpy/random + сид сэмплера и worker'ов DataLoader |
| Селекция чекпоинта | $(\mathrm{AUC} + \mathrm{PR\text{-}AUC})/2$ на val | порог-независимая метрика, устойчива к дисбалансу 1:2 |
| Чекпоинты | `best_model.pth` + `last_model.pth` | в чекпоинте: веса, конфиг, сид |

Точность: **bfloat16 AMP** на CUDA (диапазон экспоненты как у fp32 — GradScaler не нужен). Валидация считается в том же dtype, что и обучение. Обучение полностью воспроизводимо: фиксированный сид, детерминированный бакетный сэмплер (`set_epoch`), конфиг и сид сохраняются в чекпоинт.

Код: `model/encoder.py`, `model/heads_loss.py`, `model/metrics.py` (ROC-AUC / PR-AUC / ECE на numpy+scipy), `train_model.py`, `baselines/` (MLP и GPS энкодеры на том же контракте фронтенда).
