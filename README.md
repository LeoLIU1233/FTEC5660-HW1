# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. **How much money did I spend in total for these bills?**
2. **How much would I have had to pay without the discount?**

---

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your DeepSeek API key:

```
DEEPSEEK_API_KEY=your_key_here
```

Then run:

```bash
python3 hw1.py --image-folder public_test
cat results.csv
```

---

## Homework 1 Solution

                    ### Chain Design Visualization

```
┌──────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                │
│                                                                      │
│   public_test/receipt1.jpg                                           │
│   public_test/receipt2.jpg   ─── image_data_url() ──► base64 URLs    │
│   public_test/receipt*.jpg                                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       build_chain()                                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  ChatPromptTemplate                                         │     │
│  │                                                             │     │
│  │  [System Message]                                           │     │
│  │   • Role: precise receipt OCR calculator                    │     │
│  │   • Step 1: Read every line, list all item prices           │     │
│  │             and all discount lines                          │     │
│  │   • Step 2: Compute QUERY_1 and QUERY_2                     │     │
│  │   • QUERY_1 = final amount paid (after discounts+rounding)  │     │
│  │   • QUERY_2 = total before ALL discounts (add back every    │     │
│  │               promotion / coupon / member discount)         │     │
│  │   • Output: pure JSON only — no markdown, no explanation    │     │
│  │                                                             │     │
│  │  [User Message]                                             │     │
│  │   • image_url  ← base64-encoded receipt image               │     │
│  └──────────────────────────────┬──────────────────────────────┘     │
│                                 │  prompt | llm  (LCEL pipe)         │
│  ┌──────────────────────────────▼───────────────────────────────┐    │
│  │         ChatDeepSeek (deepseek-v4-flash-vision-exp)          │    │
│  │                     temperature = 0                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  chain returned once, reused for all
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      answer_queries()                                │
│                                                                      │
│  1. Build batch_inputs: one {"img_url": <base64>} per receipt        │
│                                                                      │
│  2. chain.batch(batch_inputs)  ── all receipts in parallel ──►       │
│     [response_1, response_2, ..., response_n]                        │
│                                                                      │
│  3. For each response:                                               │
│     ┌──────────────────────────────────────────────────────────┐     │
│     │  Strip markdown fences if present (```json ... ```)      │     │
│     │  json.loads(content)                                     │     │
│     │  → { "QUERY_1": "HK$XXX.XX", "QUERY_2": "HK$XXX.XX" }    │     │
│     │  Store as "receiptN.jpg|QUERY_1" / "receiptN.jpg|QUERY_2"│     │
│     │  On any exception → store "ERROR"                        │     │
│     └──────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         OUTPUT — results.csv                         │
│                                                                      │
│   filename     │ query   │ answer                                    │
│   ─────────────┼─────────┼────────                                   │
│   receipt1.jpg │ QUERY_1 │ HK$394.70                                 │
│   receipt1.jpg │ QUERY_2 │ HK$480.20                                 │
│   receipt2.jpg │ QUERY_1 │ HK$316.10                                 │
│   ...          │ ...     │ ...                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Solution Description

The solution uses a single-stage LangChain pipeline built with the LCEL pipe operator (`prompt | llm`). In `build_chain()`, a `ChatPromptTemplate` is constructed with two messages: a system message that instructs the model to act as a precise receipt OCR calculator and to follow a two-step process — first exhaustively listing every item price and every discount line (member discounts, percentage-off promotions, Buy-X-Save-Y deals, app coupons, etc.), then computing QUERY_1 (the final amount paid after all discounts and rounding) and QUERY_2 (the total that would have been paid without any discount, i.e. the subtotal with every discount added back) — and a user message containing the base64-encoded receipt image. The model is set to `temperature=0` to ensure deterministic, consistent extraction, and is required to return a pure JSON object with no surrounding text. In `answer_queries()`, all receipt images are processed in a single `chain.batch()` call so that API requests are parallelised, minimising total latency. Each response is then parsed with `json.loads()` after stripping any accidental markdown fences, and the resulting per-receipt answers are written to `results.csv`. The design delegates all OCR, line-item parsing, and arithmetic to the vision model in one pass per receipt, which proved sufficient to achieve 100% accuracy across all seven public test receipts.
