# 07 — Механизмы выявления галлюцинаций генеративных моделей дизайна белков

Обзор систематизирует механизмы выявления галлюцинаций: self-consistency контур (§7.2), confidence-фильтры (§7.3), оценки дизайнуемости (§7.4), биофизические фильтры (§7.5), молекулярную динамику (§7.6), специфику AlphaFold 3 (§7.7) и экспериментальную валидацию (§7.8). Связь с подходом BbValidator — в §7.9.

## 7.1 Термин: два смысла «галлюцинации»

### 7.1.1 Галлюцинация генеративной модели

Авторы AlphaFold 3 формулируют определение явно: «The biggest issue is that generative models are prone to hallucination, whereby the model may invent plausible-looking structure even in unstructured regions» [8]. В этом смысле галлюцинация — правдоподобный на вид образец, не соответствующий физически реализуемой структуре. Задокументированные примеры:

- AlphaFold2 с высоким pLDDT укладывает совершенные повторные последовательности в нереалистичные β-соленоиды с некомпенсированными заряженными остатками внутри; другие предсказатели дают для тех же последовательностей низкую уверенность или статус неупорядоченных, а МД-симуляции показывают нестабильность таких структур [13];
- AlphaFold 3 нарушает хиральность лигандов (4.4% на бенчмарке PoseBusters), порождает стерические конфликты вплоть до полного перекрытия цепей в гомомерах и воспроизводит неверные конформационные состояния [8];
- выходы галлюцинационных пайплайнов без секвенс-редизайна не сворачиваются экспериментально [10].

## 7.2 Self-consistency контур: ProteinMPNN → AF2/ESMFold → scRMSD

Центральный вычислительный механизм проверки: остов считается складываемым/дизайнуемым, если существует последовательность, рефолдящаяся обратно в этот остов. Стандартный контур:

1. для сгенерированного остова проектируют $N$ последовательностей методом inverse folding — как правило, ProteinMPNN [3];
2. каждую последовательность рефолдят предсказателем структуры: AF2 [6], ESMFold [14], RoseTTAFold2 как независимый контроль [10], либо несколькими предсказателями сразу [7];
3. рефолд сравнивают с исходным остовом после выравнивания по Кабашу.

В статье RFdiffusion критерий успеха сформулирован так: рефолд AF2 должен иметь среднее pAE < 5 и глобальный backbone-RMSD < 2 Å к дизайну; для motif-scaffolding дополнительно motif-RMSD < 1 Å; на каждый остов генерировалось 8 последовательностей ProteinMPNN [6].

MotifBench фиксирует метрику формально [14]: для остова $X_{\text{design}}$ и $k$-й спроектированной последовательности $s_k$ с предсказанной структурой $X_{\text{pred}}(s_k)$

$$\mathrm{scRMSD} = \min_{k=1,\dots,8}\ \mathrm{RMSD}_{C_\alpha}\bigl(X_{\text{design}},\ X_{\text{pred}}(s_k)\bigr),$$

остов успешен, если хотя бы одна из 8 последовательностей удовлетворяет $\mathrm{scRMSD} \le 2.0$ Å (по $C_\alpha$-атомам) и motifRMSD $\le 1.0$ Å (по атомам $N, C_\alpha, C$ мотива) [14].

| Работа | Дизайнер последовательностей | Рефолдер | $N$ секвенций | Порог успеха |
|---|---|---|---|---|
| RFdiffusion [6] | ProteinMPNN | AF2 | 8 | RMSD < 2 Å, pAE < 5 |
| Wicky et al. [10] | ProteinMPNN | AF2 + RF2 | 24–48 | pLDDT > 0.75, RMSD < 1.5 Å |
| Chroma [7] | собственная сеть дизайна | три предсказателя | 1 | TM-score, порог не фиксировался |
| MotifBench [14] | ProteinMPNN | ESMFold | 8 | scRMSD ≤ 2 Å ∧ motifRMSD ≤ 1 Å |

Известные ограничения контура:

1. **Предположение об обобщении предсказателя на новые фолды** — авторы Chroma отмечают это явно: «this sequence–structure consistency test is not perfect because it rests on the assumption that structure-prediction models will generalize to new folds and topologies» [7].
2. **Зависимость от выбора рефолдера:** счёт RFdiffusion в MotifBench составляет 28.1 с ESMFold против 22.5 с AF2 [14].
3. **Меморизация:** для мотивов, близких к нативным, ProteinMPNN+ESMFold могут «узнавать» нативную последовательность даже без подходящего скаффолда, завышая оценки [14].
4. **Стоимость:** 8 последовательностей × рефолдинг на каждый остов. При миллионах образцов контур становится узким местом пайплайна — это одна из мотиваций прямых однопроходных скореров (§7.9).

## 7.3 Confidence-метрики как фильтры галлюцинаций

pLDDT (поостаточная метрика уверенностт) и PAE (predicted aligned error) введены в AlphaFold2 [1]; pTM и ipTM для ранжирования предсказаний мономеров и комплексов — в AlphaFold-Multimer [2]. AlphaFold 3 дополнительно предсказывает матрицу ошибок расстояний (PDE) [8].

Применение в качестве фильтров генераций:

- **Пайплайн RFdiffusion для биндеров:** на каждую мишень генерировалось ~10 000 остовов, по 2 последовательности на остов (ProteinMPNN-FastRelax), затем ~20 000 дизайнов скринировались AF2 с initial guess и шаблонизацией по мишени; фильтр `pae_interaction < 10` в официальной документации назван «good predictor of a binder working experimentally», а непроходящие его дизайны не рекомендуются к заказу [18]. В основной статье отбор шёл по AF2-уверенности интерфейса и мономера [6].
- **Симметричные галлюцинации:** первичный отбор pLDDT > 0.7 и pTM > 0.7, после ProteinMPNN-редизайна — pLDDT > 0.75 и RMSD к исходному остову < 1.5 Å [10].
- **AlphaFold 3:** 25 образцов (5 сидов × 5 диффузионных выборок) ранжируются агрегатом pTM/ipTM с пенальти за стерические конфликты и нарушения хиральности лиганда [8].

Ограничения, из-за которых confidence-фильтры недостаточны сами по себе:

- AF2 присваивает высокий pLDDT физически нереалистичным β-соленоидам на повторных последовательностях [13] — высокая уверенность не гарантирует физическую правдоподобность;
- галлюцинированные неупорядоченные регионы AF3 «обычно» помечаются очень низкой уверенностью, но могут не иметь характерного лентовидного вида, который AF2 даёт в неупорядоченных регионах [8];
- MotifBench сознательно исключил confidence-критерий из протокола оценки: он «partly largely redundant with the designability metric of scRMSD < 2 Å and is highly dependent on the accuracy of confidence head» выбранного предсказателя [14].

Вывод: confidence-метрики — необходимый, но не достаточный фильтр; они отсекают грубые неудачи, но пропускают самоуверенные артефакты [13, 14].

## 7.4 Дизайнуемость через inverse folding

Дизайнуемость остова — число (или доля) последовательностей, рефолдящихся в него; практически измеряется моделями inverse folding: ESM-IF1, обученным на миллионах предсказанных структур [5], и ProteinMPNN [3].

Операционализации:

- **доля успешных остовов** — в RFdiffusion motif-scaffolding решает 23 из 25 задач бенчмарка [6], fold-обусловленная генерация даёт 42.5% (TIM-баррель) и 54.1% (NTF2) успешных образцов [6];
- **правило «хотя бы одна из $N$»** — критерий MotifBench (≥ 1 из 8 последовательностей) [14];
- **согласованность последовательность–структура** — в Chroma на каждый остов проектировалась одна последовательность собственной сетью дизайна и рефолдилась тремя предсказателями; количественный порог дизайнуемости не вводился [7].

Поучителен контрпример Chroma: авторы принципиально не фильтровали дизайны ни рефолдингом, ни энергетикой — «we deliberately did not filter designs for refolding by a structure-prediction method or using any structure–energetic calculations» [7]. Цена — консервативная экспериментальная успешность ~3% при тестировании 310 белков [7], тогда как фильтрующие пайплайны RFdiffusion дают двузначные доли успеха уже на уровне скрининга связывания [6].

## 7.5 Биофизические фильтры

Классический слой проверки:

- **энергетические функции Rosetta** [17] — в пайплайне RFdiffusion секвенс-дизайн выполняется протоколом ProteinMPNN-FastRelax [18]
- **shape complementarity (SC)** — классическая метрика качества упаковки белковых интерфейсов [16];
- **упаковка и стерика** — прямо не фильтруются, например, в Chroma (см. §7.4), что авторы компенсируют масштабом генерации и экспериментальным скринингом [7].

Практическая роль биофизических фильтров — отсеять образцы, формально проходящие self-consistency контур, но имеющие плохую геометрию интерфейса или ядра;

## 7.6 Молекулярная динамика как стресс-тест

Предсказатели структуры выдают статические модели и «typically predict static structures as seen in the PDB, not the dynamical behaviour of biomolecular systems in solution» [8]; множество случайных сидов генеративной модели не приближает ансамбль в растворе [8]. МД-симуляции поэтому используются как стресс-тест стабильности для узкого круга кандидатов, а не для скрининга.

Характерный пример именно в контексте галлюцинаций: уверенные β-соленоиды, предсказанные AF2 для повторных последовательностей, в МД-симуляциях теряют стабильность, что стало одним из доказательств их нереалистичности [13].

## 7.7 Экспериментальная валидация как золотой стандарт

Все вычислительные фильтры в конечном счёте калибруются об эксперимент:

- **Биндеры RFdiffusion:** 475 дизайнов по 5 мишеням, 19% связывателей в BLI-скрининге, аффинности до Kd = 28 нМ (HA_20), крио-ЭМ структура комплекса с RMSD 0.63 Å к дизайну; для мотива p53/MDM2 — 55/96 связывающихся дизайнов с Kd 0.5–0.7 нМ против 600 нМ у нативного пептида; для Ni²⁺-сборок — 18/36 связывают металл [6].
- **Симметричные галлюцинации [10]:** 74% редизайненных конструкций экспрессируются (медиана 247 мг/л), 7 кристаллических структур с медианным RMSD 0.6 Å к моделям, крио-ЭМ крупных колец с RMSD 0.81–2.30 Å; при этом исходные галлюцинированные последовательности экспериментально не работали — см. §7.1.
- **Chroma [7]:** 310 белков, пул-скрининг split-GFP (19/20 топ-кандидатов подтверждены вестерном против 0/20 нижних), CD/DSC с $T_m$ 64–78 °C, две кристаллические структуры с RMSD 1.0–1.1 Å к дизайнам.
- **Инверсия AF2 [12]:** лишь 7 из 39 дизайнов оказались свёрнутыми и стабильными в растворе — напоминание о том, что прохождение in silico фильтров не гарантирует успех.

## 7.8 Связь с данным проектом

BbValidator (модуль [01](01-obzor-i-postanovka.md)) реализует комплементарный перечисленным механизмам подход — прямое предсказание складываемости из геометрии остова за один проход, $X \in \mathbb{R}^{L \times 3 \times 3}$ (атомы $N, C_\alpha, C$):

1. **Прямой скоринг вместо self-consistency контура (§7.2).** Модель не требует ни дизайна последовательностей, ни рефолдинга: $P(\mathrm{Foldable})$ предсказывается непосредственно по координатам. Это ортогонально контуру ProteinMPNN→ESMFold и позволяет использовать валидатор как дешёвый префильтр миллионов остовов перед дорогими стадиями (пропускная способность — десятки–сотни цепей в секунду, модуль [05](05-eksperimenty-i-rezultaty.md), §5.5).
2. **Калиброванная уверенность вместо pLDDT/PAE (§7.3).** $P(\mathrm{Foldable})$ играет роль confidence-фильтра, но вычисляется из геометрии, а не из выхода предсказателя последовательностей; калибровка на тесте — ECE 0.0098. Уроки [13, 14] о недостаточности одних лишь confidence-метрик учтены вспомогательными головами (RMSD, стерическая плотность, тип дефекта), дающими независимые объяснения вердикта.
3. **OOD-валидация через scRMSD (§7.2).** На 18 000 скаффолдов пяти генераторов из MotifBench (RFdiffusion, RFdiffusion-AA, ODesign-Rigid, GPDL, EvoDiff — все представлены на лидерборде бенчмарка [14, 15]) предсказания значимо коррелируют с scRMSD (Спирмен до −0.46, $p \le 10^{-11}$) и корректно ранжируют генераторы по качеству (модуль [05](05-eksperimenty-i-rezultaty.md), §5.3).
4. **Детекция OOD как защита от самоуверенных артефактов (§7.3, [13]).** MC-Dropout неопределённость растёт на структурах внешних генераторов по сравнению с нативами — модель сигнализирует о выходе за пределы обучающего распределения.
5. **Биофизический сигнал встроен в признаки (§7.5).** Стерика виртуальных $C_\beta$ и водородные связи остова вычисляются физическим фронтендом (модуль 02) и подаются в модель как входы и вспомогательные таргеты.
6. **Кейс собственного генератора (§7.1.2).** Выходы пилотного чекпоинта SE(3)-диффузионной модели с 27% Ramachandran-outliers и 0.49 клэша на остаток получают $P(\mathrm{Foldable}) \approx 0$ — фильтр физически некорректных генераций работает далеко за пределами обучающего распределения (модуль [05](05-eksperimenty-i-rezultaty.md), §5.4).

Границы применимости: BbValidator не заменяет self-consistency контур, биофизические фильтры и эксперимент (§7.2–7.8) — он занимает нишу дешёвой первой стадии отбора, после которой к выжившим кандидатам применяются стандартные пайплайны верификации.

## Литература

1. Jumper J., Evans R., Pritzel A., Green T., Figurnov M., Ronneberger O., Tunyasuvunakool K., Bates R. et al. Highly accurate protein structure prediction with AlphaFold // Nature. 2021. Vol. 596, № 7873. P. 583–589. DOI: 10.1038/s41586-021-03819-2.
2. Evans R., O'Neill M., Pritzel A., Antropova N., Senior A. et al. Protein complex prediction with AlphaFold-Multimer // bioRxiv. 2021. DOI: 10.1101/2021.10.04.463034.
3. Dauparas J., Anishchenko I., Bennett N., Bai H., Ragotte R.J., Milles L.F. et al. Robust deep learning-based protein sequence design using ProteinMPNN // Science. 2022. Vol. 378, № 6615. P. 49–56. DOI: 10.1126/science.add2187.
4. Lin Z., Akin H., Rao R., Hie B., Zhu Z., Lu W., Smetanin N., Verkuil R. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model // Science. 2023. Vol. 379, № 6637. P. 1123–1130. DOI: 10.1126/science.ade2574.
5. Hsu C., Verkuil R., Liu J., Lin Z., Hie B., Sercu T., Lerer A., Rives A. Learning inverse folding from millions of predicted structures // bioRxiv. 2022. DOI: 10.1101/2022.04.10.487779.
6. Watson J.L., Juergens D., Bennett N.R., Trippe B.L., Yim J., Eisenach H.E., Ahern W., Borst A.J. et al. De novo design of protein structure and function with RFdiffusion // Nature. 2023. Vol. 620, № 7976. P. 1089–1100. DOI: 10.1038/s41586-023-06415-8.
7. Ingraham J.B., Baranov M., Costello Z., Barber K.W., Wang W. et al. Illuminating protein space with a programmable generative model // Nature. 2023. Vol. 623, № 7989. P. 1070–1078. DOI: 10.1038/s41586-023-06728-8.
8. Abramson J., Adler J., Dunger J., Evans R., Green T., Pritzel A. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3 // Nature. 2024. Vol. 630. P. 493–500. DOI: 10.1038/s41586-024-07487-w.
9. Anishchenko I., Pellock S.J., Chidyausiku T.M., Ramelot T.A., Ovchinnikov S., Hao J. et al. De novo protein design by deep network hallucination // Nature. 2021. Vol. 600. P. 547–552. DOI: 10.1038/s41586-021-04184-w.
10. Wicky B.I.M., Milles L.F., Courbet A., Ragotte R.J., Dauparas J., Kinfu E. et al. Hallucinating symmetric protein assemblies // Science. 2022. Vol. 378, № 6615. P. 56–61. DOI: 10.1126/science.add1964.
11. An L., Hicks D.R., Zorine D., Dauparas J., Wicky B.I.M., Milles L.F. et al. Hallucination of closed repeat proteins containing central pockets // Nature Structural & Molecular Biology. 2023. Vol. 30. P. 1755–1760. DOI: 10.1038/s41594-023-01112-6.
12. Goverde C.A., Wolf B., Khakzad H., Rosset S., Correia B.E. De novo protein design by inversion of the AlphaFold structure prediction network // Protein Science. 2023. Vol. 32. e4653. DOI: 10.1002/pro.4653.
13. Pratt O.S., Elliott L.G., Haon M., Mesdaghi S., Price R.M., Simpkin A.J., Rigden D.J. AlphaFold 2, but not AlphaFold 3, predicts confident but unrealistic β-solenoid structures for repeat proteins // Computational and Structural Biotechnology Journal. 2025. Vol. 27. P. 467–477. DOI: 10.1016/j.csbj.2025.01.016.
14. Zheng Z., Zhang B., Didi K., Yang K.K., Yim J., Watson J.L., Chen H.-F., Trippe B.L. MotifBench: A standardized protein design benchmark for motif-scaffolding problems // arXiv:2502.12479. 2025.
15. Alamdari S., Yang K.K. MotifBench: EvoDiff evaluation results (датасет) // Zenodo. 2025. DOI: 10.5281/zenodo.15445142.
16. Lawrence M.C., Colman P.M. Shape complementarity at protein/protein interfaces // Journal of Molecular Biology. 1993. Vol. 234, № 4. P. 946–950. DOI: 10.1006/jmbi.1993.1648.
17. Leaver-Fay A., Tyka M., Smith M.L., Lange O.F., Thompson J., Jacak R. et al. ROSETTA3: An object-oriented software suite for the simulation and design of macromolecules // Methods in Enzymology. 2011. Vol. 487. P. 545–574. DOI: 10.1016/B978-0-12-381270-4.00019-6.
18. RosettaCommons. RFdiffusion: официальный репозиторий и документация. URL: https://github.com/RosettaCommons/RFdiffusion (дата обращения: 11.08.2026).
19. Google DeepMind. AlphaFold 3: официальный репозиторий и документация. URL: https://github.com/google-deepmind/alphafold3 (дата обращения: 11.08.2026).
