"""First-run setup, portable configuration, and project lifecycle."""

import json
import sys

import pytest

from vibuild.cli import main, parser
from vibuild.config import setup, tool
from vibuild.resources import load, save


def test_unconfigured_doctor_does_not_launch_tools(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    save(config, {"runtime": {"blender": "", "codex": ""}})
    monkeypatch.setattr(
        "vibuild.install.subprocess.run",
        lambda *a, **k: pytest.fail("Missing tool launched"),
    )
    assert main(["doctor", "--config", str(config), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and {c["name"] for c in result["checks"]} == {
        "blender",
        "codex",
        "model",
    }
    assert all("fix" in c for c in result["checks"])


def test_config_paths_are_relative_to_config_not_cwd(tmp_path, monkeypatch):
    executable = tmp_path / "bin/blender"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    config = tmp_path / "vibuild.json"
    save(config, {"runtime": {"blender": "bin/blender"}})
    monkeypatch.chdir(tmp_path.parent)
    assert tool(config, "blender") == str(executable)


def test_python_tool_paths_preserve_virtual_environment(tmp_path):
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    config = tmp_path / "vibuild.json"
    save(
        config,
        {
            "runtime": {
                "vidmap": {"source": "third_party/vidmap", "python": ".venv/bin/python"}
            }
        },
    )
    assert tool(config, "vidmap.python") == str(executable)


def test_setup_updates_explicit_values_without_overwriting_existing_tools(tmp_path):
    config = tmp_path / "vibuild.json"
    setup(config, blender="/chosen/blender", model="first")
    setup(config, model="second")
    value = load(config)
    assert (
        value["runtime"]["blender"] == "/chosen/blender"
        and value["session"]["model"] == "second"
    )


def test_project_init_and_status_work_without_model_or_registration_tools(
    tmp_path, capsys
):
    config = tmp_path / "local.json"
    save(config, {"runtime": {"blender": "", "codex": ""}})
    project = tmp_path / "my-scene"
    assert main(["init", str(project), "--config", str(config)]) == 0
    assert (project / "project.json").exists() and not (
        project / "scene.blend"
    ).exists()
    assert main(["status", str(project), "--json"]) == 0
    state = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert state["state"] == "new"
    assert main(["pause", str(project)]) == 0
    assert main(["status", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["paused"]


@pytest.mark.parametrize(
    "name", ["tool", "task", "start", "workflow", "workflow-task", "report", "check"]
)
def test_removed_entrypoints_have_no_compatibility_aliases(name):
    with pytest.raises(SystemExit):
        parser().parse_args([name])
