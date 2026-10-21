# Paper 7 — исследовательский код v0.1

**Цель:** отличить задержку начала децидуализации от изменения амплитуды гормонального ответа по повторным транскриптомным измерениям стромы. Это прототип; реальные GEO-результаты здесь ещё НЕ рассчитаны. Никаких клинических или причинных выводов.

## Источники

- [GSE243039](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243039) — основной: 4 control и 4 endometriosis, EPC, дни 0/2/4/6/8.
- [GSE246591](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246591) — внешний временной референс 0–96 часов, отличный протокол; НЕ объединять с основным.
- [GSE127918](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE127918) — single-cell, отдельно в следующем этапе.

## 1. Установка и тест (без интернета)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m paper7.cli simulate --out results/synthetic --reps 50
```

Последняя команда создаёт ТОЛЬКО синтетические результаты. Их нельзя использовать как доказательства для статьи.

## 2. GEO-аудит и контрольные траектории

```bash
python paper7_pilot.py --mode metadata --out geo_results
python paper7_pilot.py --mode dynamics --out geo_results
```

Для офлайн-запуска обработанные матрицы можно загрузить вручную по ссылкам:

- [GSE243039_normalized_all_counts.csv.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243039/suppl/GSE243039_normalized_all_counts.csv.gz) -> geo_results/raw/GSE243039/
- [GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE246nnn/GSE246591/suppl/GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz) -> geo_results/raw/GSE246591/

Метаданные читаются из заголовков GEO Series Matrix; при невозможности однозначно связать столбец матрицы с GSM скрипт останавливается и требует ручной файл сопоставления. GSE246591 пока имеет статус донора reference_status_unverified.

## 3. Подтвердить метаданные, заморозить гипотезу, выполнить сравнение

```bash
python -m paper7.cli prepare-review --results geo_results --out geo_results/metadata/GSE243039_REVIEW.csv
```

В review CSV нужно **вручную** проверить все 40 GSM по публикации/GEO: поля reviewed, include, donor_id, group, days, treatment, review_provenance, а у исключённых записать exclusion_reason. Подсказки suggested_*_UNVERIFIED НЕ являются подтверждением.

Следующий блок намеренно требует отдельно проверенного набора генов. Файл config/programs_PROVISIONAL.gmt — демонстрационный, его нельзя использовать для финальных биологических гипотез. Создай самостоятельно проверенный config/programs_CURATED.gmt (путь/генные программы с обоснованием и источниками) **до просмотра пациентских кривых**.

```bash
python -m paper7.cli freeze \
  --results geo_results \
  --review geo_results/metadata/GSE243039_REVIEW.csv \
  --gmt config/programs_CURATED.gmt \
  --out geo_results/frozen_design.json \
  --curation-note "Programs independently verified against biological pathway resources before case fitting."

python -m paper7.cli fit \
  --results geo_results \
  --review geo_results/metadata/GSE243039_REVIEW.csv \
  --gmt config/programs_CURATED.gmt \
  --lock geo_results/frozen_design.json \
  --out results/real_stage1
```

Опция --online у команды fit разрешает скачать отсутствующую обработанную матрицу. Изменение review/GMT/метаданных после freeze блокируется проверкой SHA-256; новый план сохраняйте как отдельный lock.

## Что будет на выходе fit

- tables/donor_program_scores.csv — scores с нормализацией по контролям, без подгонки под исходы.
- tables/donor_model_comparisons.csv — amplitude-only vs amplitude+onset-delay, held-out-timepoint RMSE по каждому донору.
- tables/control_local_identifiability.csv — корреляция, обусловленность локального Якобиана PCHIP/spline.
- tables/CONTROL_ONLY_exploratory_sampling_designs.csv — исследовательские варианты временных точек (контроли).
- tables/case_sensitivity_to_control_reference.csv — чувствительность к исключению контрольного донора.
- tables/descriptive_group_summary_NO_PVALUES.csv — только описательная статистика.
- tables/case_lag_RSS_profiles_NOT_CI.csv — индивидуальные профили ошибки для задержки (НЕ доверительные интервалы).
- figures/*.png — динамика по донорам.
- run_manifest.json — provenance и SHA-256.

## Ограничения

1. 5 временных точек одного донора НЕ равны 5 независимым женщинам.
2. LOTO по времени — прогноз измерения внутри донора, НЕ независимая внешняя проверка.
3. Параметр задержки ограничен 0–2 днями и означает задержку начала in vitro ответа, не истинную фазу менструального цикла.
4. Высокий остаток после двух моделей не доказывает «перестройку» сети.
5. Локальная идентифицируемость не равна глобальной; GSE246591 не объединяется с EPC-культурой.
6. Тесты против SITAR/Genes2Genes, независимая межпротокольная валидация и публикационный статистический анализ — последующие этапы.

Подробнее: METHODS.md.
