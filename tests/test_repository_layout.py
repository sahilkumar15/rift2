from pathlib import Path


def test_scripts_directory_contains_only_shell_launchers():
    root = Path(__file__).resolve().parents[1]
    files = [p for p in (root / "scripts").iterdir() if p.is_file()]
    assert files
    assert all(p.suffix == ".sh" for p in files)


def test_no_old_rift_namespace_package():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "rift").exists()
