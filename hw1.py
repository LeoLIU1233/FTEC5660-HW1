import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import json

load_dotenv()

# Provided helper function (DO NOT EDIT)
def image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once."""
    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise receipt OCR calculator for Hong Kong supermarket receipts.\n"
         "Step 1: Read EVERY line on receipt, list all item original prices and all discount lines (including Buy X Save Y, member 5% OFF, App discount, item markdowns).\n"
         "Step 2: Calculate QUERY_1 and QUERY_2 strictly based on the full list you extracted.\n\n"
         "QUERY_1: Final amount paid after ALL discounts and rounding. Output with HK$, 2 decimal places.\n"
         "QUERY_2: Total original price BEFORE ALL discounts. Add back EVERY discount. Sum all original item prices without any promotions. Output with HK$, 2 decimal places.\n\n"
         "Return ONLY a valid JSON object with keys QUERY_1 and QUERY_2. Do NOT add any extra text, markdown, or explanation. Only pure JSON."),
        ("user", [{"type": "image_url", "image_url": {"url": "{img_url}"}}])
    ])
    chain = prompt | llm
    return chain



def answer_queries(chain: Any, images: list[Path]) -> dict[str, str]:
    """Process all receipts, return dict keyed by filename + query name."""
    batch_inputs = []
    for img_path in images:
        img_url = image_data_url(img_path)
        batch_inputs.append({"img_url": img_url})

    responses = chain.batch(batch_inputs)
    result = {}

    for img_path, resp in zip(images, responses):
        try:
            # 清理可能的 markdown 包裹，只提取 JSON
            content = resp.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            parsed = json.loads(content)
            result[f"{img_path.name}|QUERY_1"] = parsed["QUERY_1"]
            result[f"{img_path.name}|QUERY_2"] = parsed["QUERY_2"]
        except Exception:
            result[f"{img_path.name}|QUERY_1"] = "ERROR"
            result[f"{img_path.name}|QUERY_2"] = "ERROR"
    return result



# ========== BELOW IS PROVIDED RUNNER / SCORING CODE. DO NOT EDIT ==========
import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, required=True)
    args = parser.parse_args()
    image_folder = Path(args.image_folder)
    image_paths = sorted([p for p in image_folder.glob("receipt*.jpg")])

    chain = build_chain()
    answers = answer_queries(chain, image_paths)

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "query", "answer"])
        for img in image_paths:
            writer.writerow([img.name, "QUERY_1", answers.get(f"{img.name}|QUERY_1", "")])
            writer.writerow([img.name, "QUERY_2", answers.get(f"{img.name}|QUERY_2", "")])
    print(f"Processed {len(image_paths)} receipt(s). Wrote results.csv")

if __name__ == "__main__":
    main()
