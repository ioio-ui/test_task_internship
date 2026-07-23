import argparse
import html

import gradio as gr

from inference import build_ner_pipeline, extract_mountains


def highlight_html(text: str, mountains: list[dict]) -> str:
    if not mountains:
        return html.escape(text)
    parts = []
    prev = 0
    for m in sorted(mountains, key=lambda x: x["start"]):
        parts.append(html.escape(text[prev:m["start"]]))
        score = m["score"]
        word = html.escape(text[m["start"]:m["end"]])
        parts.append(
            f'<mark style="background:#ffe680;border-radius:4px;padding:1px 4px;" '
            f'title="score={score:.3f}">{word}</mark>'
        )
        prev = m["end"]
    parts.append(html.escape(text[prev:]))
    return "".join(parts)


def build_results_table(mountains: list[dict]) -> str:
    if not mountains:
        return "<p style='color:#888'>No mountain names detected.</p>"
    rows = "".join(
        f"<tr><td><b>{html.escape(m['word'])}</b></td>"
        f"<td>{m['start']}–{m['end']}</td>"
        f"<td>{'█' * int(m['score'] * 20)} {m['score']:.3f}</td></tr>"
        for m in mountains
    )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='border-bottom:1px solid #ccc'>"
        "<th align='left'>Mountain</th><th align='left'>Chars</th><th align='left'>Confidence</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def predict(text: str, ner_pipeline) -> tuple[str, str]:
    if not text.strip():
        return "<p style='color:#888'>Enter a sentence above.</p>", ""
    mountains = extract_mountains(text, ner_pipeline)
    highlighted = f"<p style='font-size:1.1em;line-height:1.8'>{highlight_html(text, mountains)}</p>"
    return highlighted, build_results_table(mountains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_output")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from '{args.model}' …")
    ner = build_ner_pipeline(args.model)
    print("Model ready.\n")

    examples = [
        "Last summer, our team attempted to summit Mount Everest but turned back due to weather.",
        "The documentary features stunning aerial footage of K2 and Kangchenjunga.",
        "Mont Blanc stands at 4808 meters above sea level.",
        "Our journey took us past the Matterhorn and through the valleys below Mont Blanc.",
        "The Amazon River runs through South America.",
        "She bought a new laptop for her studies.",
    ]

    with gr.Blocks(title="Mountain NER") as demo:
        gr.Markdown("## Mountain Named Entity Recognition\nType any sentence to detect mountain names.")
        with gr.Row():
            txt = gr.Textbox(
                label="Input sentence",
                placeholder="e.g. We climbed Mount Everest and K2 last summer.",
                lines=3,
                scale=4,
            )
            btn = gr.Button("Detect", variant="primary", scale=1)
        highlighted_out = gr.HTML(label="Annotated text")
        table_out = gr.HTML(label="Detected entities")
        gr.Examples(examples=examples, inputs=txt)
        btn.click(fn=lambda t: predict(t, ner), inputs=txt, outputs=[highlighted_out, table_out])
        txt.submit(fn=lambda t: predict(t, ner), inputs=txt, outputs=[highlighted_out, table_out])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
