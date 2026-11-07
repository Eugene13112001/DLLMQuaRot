# Сжатие KV-кэша блочных диффузионных моделей — статья 1

Код этапа 1.1: прогон без сжатия на GSM8K для SDAR-8B (блоки 4/8/16/32/64) и Fast-dLLM v2 (блок 32).

## Что внутри

- `blockkv/decode.py` — свои циклы генерации, повторяющие родные:
  - SDAR — `block_diffusion_generate` из JetAstra/SDAR, стратегия `low_confidence_dynamic`, жадный выбор;
  - Fast-dLLM v2 — `generate()` из modeling.py, temperature 0, без block cache.

  Всё, что попадает в кэш (префилл и каждый готовый блок), проходит через `on_store(cache, start, end)`. Туда на шаге 1.2 подключается сжатие. Текущий блок в кэш не попадает.
- `blockkv/run_gsm8k.py` — одна конфигурация на GSM8K. На каждую задачу пишется одна строка jsonl: решена ли, длина ответа, число шагов, число forward-проходов, вынужденные записи, сколько токенов записано на каждом шаге. Прерванный запуск продолжается с того же места.
- `blockkv/summarize.py` — таблица «блок → точность, длина, шаги» и парное сравнение двух запусков (`--compare`: совпадение ответов и разница на одних и тех же задачах).
- `scripts/stage1_baseline.sh` — все конфигурации этапа 1.1.

## Модели (скачать в `$MODELS_DIR`, имена папок как на HF)

| Конфигурация | Репозиторий |
|---|---|
| SDAR-8B, блок 4 | `JetLM/SDAR-8B-Chat` |
| SDAR-8B, блоки 8/16/32/64 | `JetLM/SDAR-8B-Chat-b{8,16,32,64}` |
| Fast-dLLM v2 | `Efficient-Large-Model/Fast_dLLM_v2_7B` |
| SDAR-30B (этап 6) | `JetLM/SDAR-30B-A3B-Chat`, `...-Chat-b{8,16,32,64}` |

GSM8K: `openai/gsm8k` (main, test) или `test.jsonl` из grade-school-math, путь передаётся в `GSM8K`.

## Запуск

```bash
pip install -r requirements.txt               # transformers==4.53.1, as Fast-dLLM v2 pins it
pip install flash-attn --no-build-isolation   # SDAR imports it
export MODELS_DIR=/path/to/models GSM8K=/path/to/gsm8k/test.jsonl GPUS="0 1 2 3"
N=20 bash scripts/stage1_baseline.sh            # проверка: читать ответы в out/stage1/20/*.jsonl
N=200 REPEAT=1 bash scripts/stage1_baseline.sh  # + проверка детерминизма
bash scripts/stage1_baseline.sh                 # все 1319
```

По умолчанию 12 конфигураций: SDAR-8B × 5 блоков и Fast-dLLM v2 (блок 32), каждая с порогом 1,0 (сверка с авторами и SoloQ) и с порогом 0,9 (`*_t0.9.jsonl`, точка отсчёта для всех опытов со сжатием). Свой список — `RUNS="sdar8b:16 sdar8b:16:0.9"` (модель:блок[:порог]).

С чем сверять (без сжатия, GSM8K):

| Конфигурация | Опубликовано | Где |
|---|---|---|
| SDAR-8B, блок 4, порог 1,0 | 91,3 | SDAR, Table 2 (жадный, по одному токену за шаг) |
| SDAR-8B, блоки 8–64 | только график | SDAR, Figure 3 (там же пороги 0,80–0,95) |
| Fast-dLLM v2, блок 32, порог 1 | 84,2 | SoloQ; BitSieve ~84 |

Для Fast-dLLM v2 на первых трёх задачах ответы нашего цикла сверяются с родным `model.generate()` токен в токен (`--check-native`).

Настройки по умолчанию — как в оценке авторов:

| | SDAR (OpenCompass `huggingface_bd3.py`, `eval_sdar_hf.py`) | Fast-dLLM v2 (`v2/eval.py`, `eval_script.sh`) |
|---|---|---|
| Порог | 1,0 (по одному токену за шаг) | 1 |
| Шагов на блок / подблок | = размер блока | блок 32, подблок 8 |
| Длина ответа | до 4096 | до 2048 |
| Тип чисел | float16 | bfloat16 |
| Стоп-токены | `generation_config` + eos токенизатора (151645, 151643) | только `<|im_end|>` (151645), как в их `generate()` |
| Кодирование промпта | `add_special_tokens=False` | по умолчанию |

Для режима с порогом из плана — `--threshold 0.9`.

Промпт у каждой модели свой, как в оценке авторов (`--prompt-style` меняет):

| Модель | Сообщение пользователя | Системное |
|---|---|---|
| SDAR (OpenCompass `gsm8k_0shot_v2_gen_17d799`) | `{question}\nPlease reason step by step, and put your final answer within \boxed{}.` | нет |
| Fast-dLLM v2 (lm-eval `gsm8k` 0-shot + их `eval.py`) | `Question: {question}\nPlease reason step by step, and put your final answer within \boxed{{}}.` — двойные скобки так и есть у них | «You are a helpful assistant.» — шаблон чата добавляет сам |
## Этап 1.2: сжатие KV-кэша

`blockkv/kvquant.py` — квантизация в духе KIVI, подключается в `on_store`: динамическая, несимметричная, min-max, округление к ближайшему. Это fake-quant: кэш квантуется и тут же восстанавливается на месте, так что модель читает ровно то, что вернул бы настоящий 2/3/4-битный кэш.

- **Вариант A (буфер).** Готовые блоки лежат в кэше в 16 битах, пока не наберётся группа из 32 токенов. Затем группа квантуется один раз и больше не меняется. Гранулярность одна для любого размера блока, поэтому честных бит всегда `биты + 32/G` (2 бита при G = 32 → 3).
- **Ключи** квантуются по каналам (одна пара min/шаг на канал на 32 токена), **значения** — по токенам (на 32 канала токена).
- Внутри всё считается во float32. Min и шаг перед восстановлением округляются до типа модели, как если бы они хранились рядом с кодами.
- Текущий блок в кэш не попадает и поэтому не сжимается никогда.

Параметры `run_gsm8k`: `--kv-bits {2,3,4,8}`, `--kv-group` (32), `--kv-window` (0 — сколько последних токенов всегда держать в 16 битах), `--kv-sink` (0), `--key-mode {channel,token}`. Для поиска настроек — `--subset 400 --seed 0`: фиксированные 400 случайных задач, всегда одни и те же.

```bash
bash scripts/stage1_kv.sh                  # поиск: 400 задач, out/stage1_kv/400/
SUBSET= bash scripts/stage1_kv.sh          # итоговые конфигурации: все 1319
python -m blockkv.summarize --compare out/stage1/full/sdar8b_b16_t0.9.jsonl out/stage1_kv/400/sdar8b_b16_t0.9_kv2.jsonl
```

Сравнение идёт по общим задачам, так что базовый прогон на 1319 задачах подходит и для подвыборки. В строке каждой задачи записаны `kv_relerr_k`/`kv_relerr_v` (относительная ошибка квантования ключей и значений) и `kv_bits_avg` (честных бит в среднем по кэшу с учётом 16-битного хвоста).
