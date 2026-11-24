"""Unit tests for others.setup_stockfish and missing-Stockfish startup warning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch
import urllib.error
import zipfile

import main
from others import setup_stockfish


def test_verify_sha256_accepts_match_and_deletes_mismatch(tmp_path: Path) -> None:
    good = tmp_path / "good.bin"
    good.write_bytes(b"stockfish-test-bytes")
    expected = hashlib.sha256(b"stockfish-test-bytes").hexdigest()

    ok, actual = setup_stockfish.verify_sha256(good, expected)
    assert ok is True
    assert actual == expected
    assert good.is_file()

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"corrupted-bytes")
    ok_bad, actual_bad = setup_stockfish.verify_sha256(bad, expected)
    assert ok_bad is False
    assert actual_bad != expected
    assert not bad.exists(), "Corrupted file must be deleted on hash mismatch"


def test_ensure_stockfish_noop_when_already_present(tmp_path: Path) -> None:
    target = tmp_path / "stockfish.exe"
    target.write_bytes(b"existing-binary")
    with patch.object(setup_stockfish, "_download_with_progress") as dl:
        assert setup_stockfish.ensure_stockfish(target) is True
        dl.assert_not_called()


def test_ensure_stockfish_handles_network_failure_with_manual_usb_hint(
    tmp_path: Path, capsys
) -> None:
    target = tmp_path / "engine" / "stockfish.exe"
    with patch.object(
        setup_stockfish,
        "_download_with_progress",
        side_effect=urllib.error.URLError("School firewall blocked host"),
    ):
        assert setup_stockfish.ensure_stockfish(target) is False
    assert not target.exists()
    out = capsys.readouterr().out
    assert "USB" in out
    assert setup_stockfish.STOCKFISH_RELEASE_URL in out


def test_ensure_stockfish_extracts_and_verifies_both_hashes(tmp_path: Path) -> None:
    target = tmp_path / "engine" / "stockfish.exe"
    fake_exe_bytes = b"MZ-fake-stockfish-19-binary"
    fake_exe_hash = hashlib.sha256(fake_exe_bytes).hexdigest()

    archive_src = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_src, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("stockfish/stockfish-windows-x86-64-universal.exe", fake_exe_bytes)
    archive_bytes = archive_src.read_bytes()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()

    def _fake_download(_url: str, dest: Path, timeout: int = 30) -> None:
        dest.write_bytes(archive_bytes)

    with patch.object(setup_stockfish, "_download_with_progress", side_effect=_fake_download), patch.object(
        setup_stockfish, "STOCKFISH_ARCHIVE_SHA256", archive_hash
    ), patch.object(setup_stockfish, "STOCKFISH_BINARY_SHA256", fake_exe_hash):
        assert setup_stockfish.ensure_stockfish(target) is True

    assert target.is_file()
    assert target.read_bytes() == fake_exe_bytes


def test_ensure_stockfish_aborts_and_deletes_on_archive_hash_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "engine" / "stockfish.exe"

    def _fake_download(_url: str, dest: Path, timeout: int = 30) -> None:
        dest.write_bytes(b"tampered-zip-payload")

    with patch.object(setup_stockfish, "_download_with_progress", side_effect=_fake_download):
        assert setup_stockfish.ensure_stockfish(target) is False
    assert not target.exists()


def test_startup_self_check_warns_when_stockfish_missing() -> None:
    with patch.object(main, "get_stockfish_path", return_value=None):
        warnings = main._run_startup_self_check()
    assert any("Stockfish" in w for w in warnings)
