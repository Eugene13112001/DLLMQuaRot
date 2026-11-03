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
pip install -r requirements.txt
export MODELS_DIR=/path/to/models GSM8K=/path/to/gsm8k/test.jsonl GPUS="0 1 2 3"
N=20 bash scripts/stage1_baseline.sh            # проверка: читать ответы в out/stage1/20/*.jsonl
N=200 REPEAT=1 bash scripts/stage1_baseline.sh  # + проверка детерминизма
bash scripts/stage1_baseline.sh                 # все 1319
```

Для Fast-dLLM v2 на первых трёх задачах ответы нашего цикла сверяются с родным `model.generate()` токен в токен (`--check-native`).

Настройки по умолчанию: SDAR — порог 0,9, шагов на блок = размер блока; Fast-dLLM v2 — блок 32, подблок 8, порог 1 (как в SoloQ и в их `eval_script.sh`); до 2048 токенов ответа.

Промпт у каждой модели свой, как в оценке авторов (`--prompt-style` меняет):

| Модель | Сообщение пользователя | Системное |
|---|---|---|
| SDAR (OpenCompass `gsm8k_0shot_v2_gen_17d799`) | `{question}\nPlease reason step by step, and put your final answer within \boxed{}.` | нет |
| Fast-dLLM v2 (lm-eval `gsm8k` 0-shot + их `eval.py`) | `Question: {question}\nPlease reason step by step, and put your final answer within \boxed{{}}.` — двойные скобки так и есть у них | «You are a helpful assistant.» — шаблон чата добавляет сам |

Опубликованные цифры SDAR посчитаны с порогом 1,0 (по одному токену за шаг) и до 4096 токенов. Для сверки с их таблицей запускать `--threshold 1.0 --gen-length 4096`.
