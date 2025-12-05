from pathlib import Path

# REPO_ROOT -> project root (two parents above src/utils/paths.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_path(*parts) -> Path:
    """
    Return an absolute Path inside the repository.
    Example: data_path('exercise_1', 'images', 'img.png')
    """
    return REPO_ROOT.joinpath(*parts)


def data_path_str(*parts) -> str:
    return str(data_path(*parts))
