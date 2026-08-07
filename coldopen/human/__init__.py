"""Ingestion of real player data, for the transfer half of the project.

Everything here reads public, rank-labelled records from games that already run
their own rating systems. It is used to *test* methods built without it, never
to build them: a method that needs the target game's human telemetry to be
configured has not solved the cold-start problem, it has relocated it.

Player identifiers are hashed at the boundary. The pipeline needs a stable key
and a skill label; it never needs a name, and an exported artefact carrying one
would be a liability with no scientific value attached.
"""

from coldopen.human.axes import standardise, tetrio_axes

#: Each source knows how to reduce itself to the common (speed, efficiency,
#: skill) representation. Adding a game is a row here and an ingestion module.
LOADERS = {
    "tetrio": lambda blob: tetrio_axes(blob["rounds"], blob["labels"]),
}


def load_game(name, blob):
    """Standardised rows for one ingested game, or [] if it is not supported."""
    loader = LOADERS.get(name)
    if loader is None:
        return []
    return standardise(loader(blob))


def load_csv_game(name, path):
    """Sources that ship as a flat file rather than an API response."""
    if name == "skillcraft":
        from coldopen.human.skillcraft import skillcraft_axes
        return standardise(skillcraft_axes(path))
    return []
