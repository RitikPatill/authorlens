"""End-to-end demo: run the full AuthorLens pipeline on the bundled dataset.

Usage:
    python demo.py
"""

import json
import sys
from pathlib import Path

# Allow running from repo root without editable install
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.table import Table

from authorlens import scorer

console = Console()


def load_pairs(labels_path: str) -> list[dict]:
    with open(labels_path, encoding="utf-8") as f:
        return json.load(f)


def run_demo() -> None:
    pairs = load_pairs("data/demo/labels.json")

    table = Table(
        title="AuthorLens — Demo Dataset Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Pair", justify="right", style="dim", width=5)
    table.add_column("Topic", min_width=22)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Embed Sim", justify="right", width=10)
    table.add_column("Stylo Sim", justify="right", width=10)
    table.add_column("Verdict", min_width=24)
    table.add_column("✓", justify="center", width=3)

    results = []
    for p in pairs:
        human_text = Path(p["human"]).read_text(encoding="utf-8")
        gpt4_text = Path(p["gpt4"]).read_text(encoding="utf-8")

        score = scorer.fuse(human_text, gpt4_text)
        v = scorer.verdict(score.overall)
        correct = v == "likely different authors"
        results.append({"id": p["id"], "score": score.overall, "correct": correct})

        checkmark = "[green]✓[/green]" if correct else "[red]✗[/red]"
        score_color = "green" if score.overall < 0.4 else ("yellow" if score.overall < 0.7 else "red")

        table.add_row(
            str(p["id"]),
            p["topic"],
            f"[{score_color}]{score.overall:.3f}[/{score_color}]",
            f"{score.embedding_sim:.3f}",
            f"{score.stylometric_sim:.3f}",
            v,
            checkmark,
        )

    console.print()
    console.print(table)
    console.print()

    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / len(results)
    acc_color = "green" if accuracy >= 0.8 else "yellow"
    console.print(
        f"[bold]Accuracy on demo set:[/bold] "
        f"[{acc_color}]{accuracy:.0%}[/{acc_color}] "
        f"({n_correct}/{len(results)} correct)"
    )
    console.print()


if __name__ == "__main__":
    run_demo()
