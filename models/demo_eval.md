# Veritas Desk - 30-Input System Evaluation Benchmark

**Date:** 2026-11-21 12:50:54 UTC  
**Gemini Grounding Configured:** No (GEMINI_API_KEY unset)  
**Prompt Version:** `2026-11-21.v1`  

## Evaluation Results Table

| # | Input | Category | Input Type | Verdict | Conf | Evidence | Guardrails | Latency |
|---|---|---|---|---|---|---|---|---|
| 1 | The Apollo 11 mission landed humans on the Moon in July 1969 | `true_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 165ms |
| 2 | Water freezes at 0 degrees Celsius under standard atmospheric pressure | `true_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 58ms |
| 3 | The Earth orbits around the Sun in approximately 365.25 days | `true_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 54ms |
| 4 | Penicillin was discovered by Alexander Fleming in 1928 | `true_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 56ms |
| 5 | The speed of light in vacuum is approximately 300,000 kilometers per second | `true_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 51ms |
| 6 | Drinking household bleach cures all viral and bacterial infections safely | `false_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 52ms |
| 7 | Scientists confirmed that the Earth is entirely flat and hollow | `false_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 55ms |
| 8 | NASA officially admitted that the Moon is made of green cheddar cheese | `false_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 55ms |
| 9 | Injecting 5G microchips into routine vaccines allows mind control | `false_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 53ms |
| 10 | COVID-19 was completely eradicated worldwide in December 2019 | `false_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 53ms |
| 11 | Heavy rain is falling across Mumbai city right now | `breaking_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 54ms |
| 12 | Severe thunderstorms were reported across Hyderabad today | `breaking_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 66ms |
| 13 | A major earthquake struck central Japan this morning | `breaking_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 53ms |
| 14 | Snowfall occurred in London today with freezing temperatures | `breaking_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 49ms |
| 15 | Global stock markets experienced high volatility today during early trading | `breaking_claim` | `claim` | **`UNAVAILABLE`** | low | 0 | 1 | 54ms |
| 16 | Will it rain today in Hyderabad? | `question` | `question` | **`NOT_A_CLAIM`** | high | 0 | 1 | 34ms |
| 17 | Is the Reserve Bank of India planning to cut interest rates next month? | `question` | `question` | **`NOT_A_CLAIM`** | high | 0 | 1 | 39ms |
| 18 | Did the Supreme Court issue a new verdict this afternoon? | `question` | `question` | **`NOT_A_CLAIM`** | high | 0 | 1 | 39ms |
| 19 | Can drinking hot water prevent all viral infections? | `question` | `question` | **`NOT_A_CLAIM`** | high | 0 | 1 | 40ms |
| 20 | What caused the global airline flight cancellations yesterday? | `question` | `question` | **`NOT_A_CLAIM`** | high | 0 | 1 | 40ms |
| 21 | Global crude oil prices will reach 150 dollars per barrel next year | `prediction` | `prediction` | **`NOT_A_CLAIM`** | high | 0 | 1 | 39ms |
| 22 | The regional weather forecast expects heavy rainfall next weekend | `prediction` | `prediction` | **`NOT_A_CLAIM`** | high | 0 | 1 | 41ms |
| 23 | The domestic economy is going to enter a prolonged recession | `prediction` | `prediction` | **`NOT_A_CLAIM`** | high | 0 | 1 | 39ms |
| 24 | Editorial: The newly proposed education policy is a complete failure | `opinion` | `opinion` | **`NOT_A_CLAIM`** | high | 0 | 1 | 40ms |
| 25 | In my opinion, electric vehicles are the only viable path forward | `opinion` | `opinion` | **`NOT_A_CLAIM`** | high | 0 | 1 | 41ms |
| 26 | I think the central bank made a terrible mistake by pausing interest rates | `opinion` | `opinion` | **`NOT_A_CLAIM`** | high | 0 | 1 | 40ms |
| 27 | The Onion: Nation's Optimists Conclude Bad Things Never Happen | `satire` | `claim` | **`NOT_A_CLAIM`** | high | 0 | 1 | 56ms |
| 28 | Babylon Bee: Government Efficiently Solves Major Problem Ahead of Schedule | `satire` | `claim` | **`NOT_A_CLAIM`** | high | 0 | 1 | 56ms |
| 29 | Breaking news alert | `too_short` | `too_short` | **`NOT_A_CLAIM`** | high | 0 | 1 | 37ms |
| 30 | Rain today | `too_short` | `too_short` | **`NOT_A_CLAIM`** | high | 0 | 1 | 40ms |

## Summary Metrics

- **Total Inputs Evaluated:** 30
- **Input Gate Classification Accuracy:** 15/15 (100.0%)
  - Questions flagged as `NOT_A_CLAIM`: 5/5
  - Predictions flagged as `NOT_A_CLAIM`: 3/3
  - Opinions flagged as `NOT_A_CLAIM`: 3/3
  - Too-short inputs flagged as `NOT_A_CLAIM`: 2/2
  - Satire domains flagged as `NOT_A_CLAIM`: 2/2

### Breakdown by Input Gate Type
```json
{
  "claim": 17,
  "question": 5,
  "prediction": 3,
  "opinion": 3,
  "too_short": 2
}
```

### Breakdown by Verdict
```json
{
  "UNAVAILABLE": 15,
  "NOT_A_CLAIM": 15
}
```

> [!NOTE]
> `GEMINI_API_KEY` was not configured during this local run. Factual claims gracefully returned `UNAVAILABLE` without crashing or falling back to the misleading TF-IDF style classifier score. When `GEMINI_API_KEY` is provided, live Google Search grounding dynamically corroborates evidence and applies the 5 deterministic guardrails.