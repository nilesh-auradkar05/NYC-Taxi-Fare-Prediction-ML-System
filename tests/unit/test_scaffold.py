from pathlib import Path


def test_platform_v3_directories_exist() -> None:
    required_dirs = [
        "infra",
        "ingestion",
        "dbt",
        "analytics",
        "analysis",
        "notebook",
        "tests",
        "scripts",
        "outputs",
    ]

    missing = [path for path in required_dirs if not Path(path).exists()]
    assert not missing, f"Missing required platform-v3 directories: {missing}"
