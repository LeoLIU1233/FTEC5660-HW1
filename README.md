# FTEC5660 Homework 1: Receipt Chain

A LangChain pipeline that reads Hong Kong supermarket receipt images using a vision-capable DeepSeek model and answers two aggregate questions across all receipts in a folder:

1. **How much money did I spend in total for these bills?**
2. **How much would I have had to pay without the discount?**

---

## Setup

```bash
git clone <your fork's url>
cd FTEC5660
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
```

> Never commit `.env` — it is already listed in `.gitignore`.

---

## Run

```bash
python3 hw1.py --image-folder public_test
cat results.csv
```

---

## Homework 1 Solution

### Chain Design Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                  │
│                                                                         │
│   public_test/receipt1.jpg                                              │
│   public_test/receipt2.jpg  ──► image_data_url() ──► base64 data URLs   │
│   public_test/receipt*.jpg                                              │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          build_chain()                                  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ChatPromptTemplate                                               │  │
│  │                                                                   │  │
│  │  [System Message]                                                 │  │
│  │   • Role: precise receipt OCR calculator                          │  │
│  │   • Step 1 — Read every line; list all item prices                │  │
│  │             and every discount line (member %, Buy X Save Y,      │  │
│  │             App coupon, item markdowns …)                         │  │
│  │   • Step 2 — Compute and return:                                  │  │
│  │       QUERY_1 = final amount paid (after discounts + rounding)    │  │
│  │       QUERY_2 = total before ALL discounts (add every             │  │
│  │                 discount back; do NOT add back ROUNDING)          │  │
│  │   • Output: pure JSON only — no markdown, no explanation          │  │
│  │                                                                   │  │
│  │  [User Message]                                                   │  │
│  │   • image_url ← base64-encoded receipt image                      │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  │  prompt | llm   (LCEL pipe)          │
│  ┌───────────────────────────────▼───────────────────────────────────┐  │
│  │        ChatDeepSeek  ·  deepseek-v4-flash-vision-exp              │  │
│  │                      ·  temperature = 0                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │  chain returned once, reused for all
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         answer_queries()                                │
│                                                                         │
│  1. Build batch_inputs — one {"img_url": <base64>} per receipt          │
│                                                                         │
│  2. chain.batch(batch_inputs)  ── all receipts processed in parallel    │
│     → [response_1, response_2, …, response_n]                           │
│                                                                         │
│  3. For each response:                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐ │
│     │  Strip markdown fences if present  (```json … ```)              │ │
│     │  json.loads(content)                                            │ │
│     │  → { "QUERY_1": "HK$XXX.XX", "QUERY_2": "HK$XXX.XX" }           │ │
│     │  Accumulate:  total_q1 += QUERY_1                               │ │
│     │               total_q2 += QUERY_2                               │ │
│     │  On any exception → print warning, skip receipt                 │ │
│     └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  4. Return aggregate answers keyed by exact query strings:              │
│     {                                                                   │
│       "How much money did I spend …": "HK$1974.30",                     │
│       "How much would I have had to pay …": "HK$2348.20"                │
│     }                                                                   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT — results.csv                               │
│                                                                           │
│  query                                       │ model_response │correctness│
│  ────────────────────────────────────────────┼───────────────┼──────────  │
│  How much money did I spend in total …       │ HK$1974.30    │ correct    │
│  How much would I have had to pay without …  │ HK$2348.20    │ correct    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Solution Description

The solution is built as a single LangChain LCEL pipeline (`prompt | llm`) initialised once in `build_chain()`. A `ChatPromptTemplate` constructs a two-message conversation for each receipt: a system message that instructs the model to work in two explicit steps — first exhaustively enumerating every item price and every discount line on the receipt (member percentage-off, Buy-X-Save-Y promotions, app coupons, item markdowns), then computing the two query answers — and a user message containing the base64-encoded receipt image. The model (`deepseek-v4-flash-vision-exp`) is run at `temperature=0` for deterministic output and is required to return a pure JSON object with keys `QUERY_1` and `QUERY_2`, where `QUERY_1` is the final amount paid after all discounts and rounding, and `QUERY_2` is the pre-discount total with every promotion added back (but not the rounding adjustment). In `answer_queries()`, all receipt images are submitted together via `chain.batch()` so that API calls are parallelised, reducing total latency. Each response is cleaned of any accidental markdown fences before being parsed with `json.loads()`; the two monetary values are then converted to `Decimal` for precision-safe arithmetic and accumulated into running totals across all receipts. The final aggregate answers are returned as a dictionary keyed by the exact query strings, which the provided runner writes directly to `results.csv`. This design achieved 100% accuracy on all seven public test receipts.
