"""Typer CLI entry point (M4)."""

from __future__ import annotations

import itertools
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from authorlens import scorer

app = typer.Typer(name="authorlens", add_completion=False)

console = Console()

# Maps feature_deltas keys (from features.compare()) to human-readable labels
FEATURE_LABELS: dict[str, str] = {
    "ttr_dist": "Vocabulary richness (TTR)",
    "mean_sent_len_dist": "Avg sentence length",
    "std_sent_len_dist": "Sentence length variation",
    "punct_density_dist": "Punctuation density",
    "yule_k_dist": "Lexical diversity (Yule's K)",
    "burstiness_dist": "Sentence-length burstiness",
    "func_word_cosine_dist": "Function-word profile",
}

# Plain-English explanations for each feature delta
_EXPLANATIONS: dict[str, str] = {
    "ttr_dist": "One text uses notably more unique words.",
    "mean_sent_len_dist": "One text has significantly longer sentences.",
    "std_sent_len_dist": "One text varies sentence length much more.",
    "punct_density_dist": "One text uses punctuation far more heavily.",
    "yule_k_dist": "One text has markedly different lexical diversity.",
    "burstiness_dist": "One text has burstier sentence-length rhythm.",
    "func_word_cosine_dist": "Function-word usage patterns differ strongly.",
}


def _explain(key: str, delta: float) -> str:  # noqa: ARG001
    return _EXPLANATIONS.get(key, "Feature values differ notably.")


def _colour_score(value: float) -> str:
    if value >= 0.7:
        return f"[green]{value:.3f}[/green]"
    if value >= 0.4:
        return f"[yellow]{value:.3f}[/yellow]"
    return f"[red]{value:.3f}[/red]"


@app.command()
def compare(
    file_a: Path = typer.Argument(..., exists=True, dir_okay=False),
    file_b: Path = typer.Argument(..., exists=True, dir_okay=False),
    emb_weight: float = typer.Option(0.5, "--emb-weight", min=0.0, max=1.0),
    threshold: float = typer.Option(
        1.0,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Exit 1 if overall similarity exceeds this value",
    ),
) -> None:
    """Compare two text files and show stylometric similarity."""
    text_a = file_a.read_text(encoding="utf-8")
    text_b = file_b.read_text(encoding="utf-8")

    score = scorer.fuse(text_a, text_b, emb_weight)

    # Summary panel
    summary = (
        f"[bold]Overall:[/bold]        {_colour_score(score.overall)}\n"
        f"[bold]Embedding sim:[/bold]  {score.embedding_sim:.3f}\n"
        f"[bold]Stylometric sim:[/bold] {score.stylometric_sim:.3f}"
    )
    console.print(Panel(summary, title=f"[bold cyan]{file_a.name}[/] vs [bold cyan]{file_b.name}[/]"))

    # Top-3 diverging features
    top3 = sorted(score.feature_deltas.items(), key=lambda kv: kv[1], reverse=True)[:3]

    table = Table(title="Top-3 Diverging Features", show_lines=True)
    table.add_column("Feature", style="bold")
    table.add_column("Delta", justify="right")
    table.add_column("Explanation")

    for key, delta in top3:
        label = FEATURE_LABELS.get(key, key)
        explanation = _explain(key, delta)
        table.add_row(label, f"{delta:.4f}", explanation)

    console.print(table)

    if score.overall > threshold:
        raise typer.Exit(code=1)


@app.command()
def scan(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    emb_weight: float = typer.Option(0.5, "--emb-weight"),
    threshold: float = typer.Option(1.0, "--threshold"),
) -> None:
    """Scan a directory and show pairwise similarity matrix for all .txt files."""
    txt_files = sorted(directory.glob("*.txt"))

    if len(txt_files) < 2:
        typer.echo("Warning: need at least 2 .txt files to compare.")
        raise typer.Exit(code=0)

    pairs = list(itertools.combinations(txt_files, 2))
    results: list[tuple[str, str, float]] = []

    for fa, fb in pairs:
        text_a = fa.read_text(encoding="utf-8")
        text_b = fb.read_text(encoding="utf-8")
        score = scorer.fuse(text_a, text_b, emb_weight)
        results.append((fa.stem, fb.stem, score.overall))

    # Collect unique file stems in order
    stems = [f.stem for f in txt_files]

    table = Table(title="Pairwise Similarity Matrix", show_lines=True)
    table.add_column("File", style="bold")
    for stem in stems:
        table.add_column(stem, justify="center")

    # Build lookup for quick access
    score_map: dict[tuple[str, str], float] = {}
    for name_a, name_b, overall in results:
        score_map[(name_a, name_b)] = overall
        score_map[(name_b, name_a)] = overall

    for row_stem in stems:
        cells: list[str] = [row_stem]
        for col_stem in stems:
            if row_stem == col_stem:
                cells.append("[dim]—[/dim]")
            else:
                val = score_map.get((row_stem, col_stem), 0.0)
                cells.append(_colour_score(val))
        table.add_row(*cells)

    console.print(table)

    any_exceeded = any(overall > threshold for _, _, overall in results)
    if any_exceeded:
        raise typer.Exit(code=1)
