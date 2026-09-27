#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Set up the vision model and prompt once."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek
 
    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise receipt OCR calculator for Hong Kong supermarket receipts.\n"
         "Step 1: Read EVERY line on the receipt, list all item original prices and all "
         "discount lines (including Buy X Save Y, member % OFF, App discount, item markdowns).\n"
         "Step 2: Calculate QUERY_1 and QUERY_2 strictly based on the full list you extracted.\n\n"
         "QUERY_1: Final amount paid after ALL discounts and rounding. Output with HK$, 2 decimal places.\n"
         "QUERY_2: Total original price BEFORE ALL discounts. Add back EVERY discount/promotion/coupon. "
         "Do NOT add back ROUNDING. Output with HK$, 2 decimal places.\n\n"
         "Return ONLY a valid JSON object with keys QUERY_1 and QUERY_2. "
         "No extra text, no markdown fences. Only pure JSON."),
        ("user", [{"type": "image_url", "image_url": {"url": "{img_url}"}}]),
    ])
 
    return prompt | llm
 
 
def answer_queries(chain: Any, images: list[Path]) -> dict[str, str]:
    """
    Process every receipt and return the two AGGREGATE totals keyed by the
    exact query strings QUERY_1 and QUERY_2.
    """
    # ── 1. Build one input dict per image and batch-call the chain ────────────
    batch_inputs = [{"img_url": image_data_url(p)} for p in images]
    responses = chain.batch(batch_inputs)
 
    # ── 2. Parse each response and accumulate running totals ──────────────────
    total_q1 = Decimal("0")
    total_q2 = Decimal("0")
 
    for img_path, resp in zip(images, responses):
        content = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
 
        # Strip accidental markdown fences
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
 
        try:
            parsed = json.loads(content)
 
            def to_dec(val: Any) -> Decimal:
                """Convert 'HK$123.45' or 123.45 to Decimal."""
                s = str(val).replace("HK$", "").replace(",", "").strip()
                return Decimal(s).quantize(Decimal("0.01"))
 
            total_q1 += to_dec(parsed["QUERY_1"])
            total_q2 += to_dec(parsed["QUERY_2"])
 
        except Exception as exc:
            print(f"[WARN] Could not parse response for {img_path.name}: {exc}\nRaw: {content}")
 
    # ── 3. Return aggregate answers keyed by the exact query strings ──────────
    return {
        QUERY_1: f"HK${total_q1:.2f}",
        QUERY_2: f"HK${total_q2:.2f}",
    }


# ---------------------------------------------------------------------------
# Everything below is provided runner/scoring code. No edits are needed.
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")
    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")
    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")
    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
