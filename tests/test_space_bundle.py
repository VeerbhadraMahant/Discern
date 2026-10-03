import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_space_bundle", ROOT / "scripts" / "build_space_bundle.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_has_the_app_the_package_and_configs_but_no_data_or_weights(
    tmp_path: Path,
) -> None:
    out = tmp_path / "space"
    files = {p.relative_to(out).as_posix() for p in load_script().build_bundle(ROOT, out)}
    assert {"app.py", "requirements.txt", "README.md"} <= files
    assert "src/discern/serve/gradio_app.py" in files
    assert "configs/models.yaml" in files and "legacy/v0/restoration.py" in files
    assert not any(f.startswith(("data/", "weights/")) for f in files)
    assert not any(f.endswith((".pt", ".pth", ".pyc")) for f in files)


def test_bundle_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "space"
    out.mkdir()
    with pytest.raises(FileExistsError):
        load_script().build_bundle(ROOT, out)


def test_space_readme_front_matter_and_no_numbers_claimed() -> None:
    text = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    head = text.split("---")[1]
    assert "sdk: gradio" in head and "app_file: app.py" in head
    assert 'python_version: "3.12"' in head
    assert "%" not in text  # no performance figures in the description


def test_deploy_workflow_runs_only_on_version_tags() -> None:
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy-space.yml").read_text("utf-8"))
    triggers = doc.get("on", doc.get(True))  # YAML 1.1 reads the key `on` as True
    assert triggers == {"push": {"tags": ["v*"]}}


def test_clean_never_deletes_outside_the_build_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    for target in (tmp_path, tmp_path / "src"):
        target.mkdir(exist_ok=True)
        monkeypatch.setattr("sys.argv", ["x", "--clean", "--out", str(target)])
        assert script.main() == 1
    assert (tmp_path / "keep.txt").is_file() and (tmp_path / "src").is_dir()


def workflows() -> dict[str, dict[str, object]]:
    folder = ROOT / ".github" / "workflows"
    return {p.name: yaml.safe_load(p.read_text("utf-8")) for p in folder.glob("*.yml")}


def test_workflows_are_least_privilege_and_never_use_pull_request_target() -> None:
    for name, doc in workflows().items():
        triggers = doc.get("on", doc.get(True))
        assert "pull_request_target" not in triggers, name
        assert "permissions" in doc, f"{name} needs an explicit top-level permissions block"


def test_self_hosted_gate_never_runs_on_pull_requests() -> None:
    doc = workflows()["gate.yml"]
    assert "pull_request" not in doc.get("on", doc.get(True))


def test_deploy_pins_the_package_that_receives_the_token() -> None:
    text = (ROOT / ".github" / "workflows" / "deploy-space.yml").read_text("utf-8")
    assert "huggingface_hub==" in text
