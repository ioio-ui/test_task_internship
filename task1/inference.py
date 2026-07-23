import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def build_ner_pipeline(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )


def extract_mountains(text: str, ner_pipeline) -> list[dict]:
    raw = ner_pipeline(text)
    spans = [r for r in raw if r["entity_group"] == "MOUNTAIN"]
    if not spans:
        return []

    expanded = []
    for r in spans:
        start, end = r["start"], r["end"]
        while end < len(text) and text[end].isalnum():
            end += 1
        expanded.append({"start": start, "end": end, "score": float(r["score"])})

    merged: list[dict] = []
    for sp in expanded:
        if merged and sp["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            merged[-1] = {
                "start": prev["start"],
                "end":   max(prev["end"], sp["end"]),
                "score": (prev["score"] + sp["score"]) / 2,
            }
        else:
            merged.append(sp)

    return [
        {"word": text[m["start"]:m["end"]], "score": round(m["score"], 4),
         "start": m["start"], "end": m["end"]}
        for m in merged
    ]


def pretty_print(text: str, mountains: list[dict]) -> None:
    if not mountains:
        print("  (no mountain names detected)")
        return
    for m in mountains:
        bar = "█" * int(m["score"] * 20)
        print(f"  [{m['start']:3d}:{m['end']:3d}]  {m['word']:<25s}  conf={m['score']:.3f}  {bar}")
    annotated = list(text)
    for m in sorted(mountains, key=lambda x: x["start"], reverse=True):
        annotated[m["start"]:m["end"]] = list(f"**{m['word'].upper()}**")
    print("  " + "".join(annotated))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_output")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from '{args.model}' …")
    ner = build_ner_pipeline(args.model)
    print("Model ready.\n")

    if args.text:
        print(f"Input: {args.text}")
        pretty_print(args.text, extract_mountains(args.text, ner))
    else:
        print("Mountain NER — type a sentence (empty line to quit):")
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            pretty_print(text, extract_mountains(text, ner))


if __name__ == "__main__":
    main()
