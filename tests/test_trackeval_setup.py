import json
import subprocess
import sys

from motlab.evaluation import trackeval_setup
from motlab.evaluation.trackeval_setup import prepare_trackeval


def test_prepare_trackeval_without_clone_does_not_call_subprocess(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess.run should not be called when clone=False")

    monkeypatch.setattr(trackeval_setup.subprocess, "run", fake_run)

    result = prepare_trackeval(trackeval_root=tmp_path / "TrackEval", clone=False)

    assert calls == []
    assert result.cloned is False
    assert result.already_exists is False
    assert "git clone" in result.message
    assert result.setup_info_path.exists()


def test_prepare_trackeval_existing_root_does_not_clone(tmp_path):
    trackeval_root = tmp_path / "TrackEval"
    trackeval_root.mkdir()

    result = prepare_trackeval(trackeval_root=trackeval_root, clone=True)

    assert result.cloned is False
    assert result.already_exists is True
    assert result.trackeval_root == trackeval_root
    assert result.setup_info_path.exists()


def test_prepare_trackeval_with_clone_calls_git_and_records_commit(tmp_path, monkeypatch):
    trackeval_root = tmp_path / "TrackEval"
    calls = []

    def fake_run(command, check=False, capture_output=True, text=True):
        calls.append(command)
        if command[:2] == ["git", "clone"]:
            trackeval_root.mkdir()
            return subprocess.CompletedProcess(command, 0, stdout="cloned", stderr="")
        if command[:3] == ["git", "-C", str(trackeval_root)]:
            return subprocess.CompletedProcess(command, 0, stdout="abc123def456\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(trackeval_setup.subprocess, "run", fake_run)

    result = prepare_trackeval(
        trackeval_root=trackeval_root,
        repo_url="https://example.com/TrackEval.git",
        ref="main",
        clone=True,
    )

    assert ["git", "clone", "--branch", "main", "https://example.com/TrackEval.git", str(trackeval_root)] in calls
    assert result.cloned is True
    assert result.already_exists is False
    assert result.commit_hash == "abc123def456"


def test_prepare_trackeval_writes_setup_info_json(tmp_path, monkeypatch):
    trackeval_root = tmp_path / "TrackEval"

    def fake_run(command, check=False, capture_output=True, text=True):
        if command[:2] == ["git", "clone"]:
            trackeval_root.mkdir()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(trackeval_setup.subprocess, "run", fake_run)

    result = prepare_trackeval(trackeval_root=trackeval_root, clone=True)
    setup_info = json.loads(result.setup_info_path.read_text(encoding="utf-8"))

    assert setup_info["repo_url"] == result.repo_url
    assert setup_info["ref"] == result.ref
    assert setup_info["commit_hash"] == "abc123"
    assert setup_info["trackeval_root"] == str(trackeval_root)
    assert "prepared_at" in setup_info


def test_cli_prepare_trackeval_default_prints_guidance_without_clone(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "prepare-trackeval",
            "--trackeval-root",
            str(tmp_path / "TrackEval"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "cloned: False" in completed.stdout
    assert "git clone" in completed.stdout
    assert not (tmp_path / "TrackEval").exists()


def test_cli_prepare_trackeval_clone_uses_fake_local_repo(tmp_path):
    source_repo = tmp_path / "source_trackeval"
    source_repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "master"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (source_repo / "README.md").write_text("fake TrackEval\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=source_repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=source_repo, check=True, capture_output=True, text=True)

    trackeval_root = tmp_path / "TrackEval"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "prepare-trackeval",
            "--trackeval-root",
            str(trackeval_root),
            "--repo-url",
            str(source_repo),
            "--ref",
            "master",
            "--clone",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "cloned: True" in completed.stdout
    assert "commit_hash:" in completed.stdout
    assert trackeval_root.exists()


def test_gitignore_ignores_trackeval_checkout():
    gitignore = (trackeval_setup.find_project_root() / ".gitignore").read_text(encoding="utf-8")

    assert "third_party/TrackEval/" in gitignore
