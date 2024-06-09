from pathlib import Path


def increment_path(path: str, separator: str = "-") -> Path:
    path = Path(path)
    if path.exists():
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else path, ""
        for n in range(2, 9999):
            p = f"{path}{separator}{n}{suffix}"
            if not Path(p).exists():
                path = Path(p)
                break
        path.mkdir(parents=True, exist_ok=True)  # make directory
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path
