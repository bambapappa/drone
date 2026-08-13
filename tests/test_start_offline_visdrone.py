import os
import shutil
import subprocess
from pathlib import Path


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _run_launcher(tmp_path: Path, *args: str) -> list[str]:
    root = tmp_path / "project"
    scripts = root / "scripts"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(Path("scripts/start_offline_visdrone.sh"), scripts / "start_offline_visdrone.sh")
    (root / "videos").mkdir()
    (root / "videos" / "film.mp4").write_bytes(b"video")
    (root / "models").mkdir()
    (root / "models" / "visdrone-yolov8s.pt").write_bytes(b"model")

    log = tmp_path / "commands.log"
    _write_executable(
        bin_dir / "docker",
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$TEST_LOG"\nexit 0\n',
    )
    _write_executable(
        bin_dir / "python3",
        (
            '#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then exec "$TEST_PYTHON" "$@"; fi\n'
            'printf \'python3 %s\\n\' "$*" >> "$TEST_LOG"\nexit 0\n'
        ),
    )
    _write_executable(bin_dir / "curl", "#!/usr/bin/env bash\nexit 0\n")
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log),
        "TEST_PYTHON": os.sys.executable,
    }
    result = subprocess.run(
        ["bash", "scripts/start_offline_visdrone.sh", *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return log.read_text().splitlines()


def test_launcher_reuses_latest_run_in_colima_context(tmp_path):
    commands = _run_launcher(tmp_path, "videos/film.mp4")

    assert all(
        command.startswith("--context colima") for command in commands if not command.startswith("python3")
    )
    assert any("run --rm analyze /videos/film.mp4 --reuse-latest" in command for command in commands)
    assert not any(command.startswith("context use") for command in commands)


def test_launcher_fresh_flag_omits_reuse_option(tmp_path):
    commands = _run_launcher(tmp_path, "--fresh", "videos/film.mp4")

    analyze = next(command for command in commands if "run --rm analyze" in command)
    assert "--reuse-latest" not in analyze


def test_launcher_stops_safely_on_colima_vz_host_agent_error(tmp_path):
    """A failed Colima start must not progress to compose or change .env."""
    root = tmp_path / "project"
    scripts = root / "scripts"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(Path("scripts/start_offline_visdrone.sh"), scripts / "start_offline_visdrone.sh")
    (root / "videos").mkdir()
    (root / "videos" / "film.mp4").write_bytes(b"video")
    env_file = root / ".env"
    env_file.write_text("MODEL=user-choice.pt\n")

    log = tmp_path / "commands.log"
    _write_executable(
        bin_dir / "docker",
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$TEST_LOG"\nexit 1\n',
    )
    _write_executable(
        bin_dir / "colima",
        (
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$TEST_LOG"\n'
            "echo 'VZ: host agent is not running' >&2\nexit 1\n"
        ),
    )
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log),
    }
    result = subprocess.run(
        ["bash", "scripts/start_offline_visdrone.sh", "videos/film.mp4"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert 'VZ-felet "host agent is not"' in result.stderr
    assert "Starta om\nMacen" in result.stderr
    assert "Radera eller återskapa inte Colima-profilen\nautomatiskt" in result.stderr
    assert env_file.read_text() == "MODEL=user-choice.pt\n"
    assert log.read_text().splitlines() == ["--context colima info", "start"]
