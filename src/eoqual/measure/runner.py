"""
Affichage du registre des mesures (:data:`~eoqual.measure.config.MEASURES_CONFIGS`).

Exemple
-------
>>> from eoqual.measure import list_measures
>>> list_measures()                   # mesures disponibles
>>> list_measures(include_planned=True)
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .config import MEASURES_CONFIGS

__all__ = ["list_measures"]


def list_measures(include_planned: bool = False) -> None:
    """Affiche les mesures disponibles (et prévues si ``include_planned``)."""
    console = Console()
    table = Table(title="Mesures de eoqual.measure", header_style="bold magenta")
    table.add_column("Mesure", style="bold cyan", no_wrap=True)
    table.add_column("Famille")
    table.add_column("Statut")
    table.add_column("Point d'entrée")
    table.add_column("Grandeurs produites")

    for name, cfg in MEASURES_CONFIGS.items():
        if cfg["status"] != "available" and not include_planned:
            continue
        status = "[green]disponible" if cfg["status"] == "available" else "[yellow]prévue"
        table.add_row(name, cfg["family"], status, cfg["entry"], cfg["outputs"])

    console.print(table)
