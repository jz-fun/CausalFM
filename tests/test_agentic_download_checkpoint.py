"""Network-free tests of checkpoint integrity and destination preservation."""

import hashlib
import io

import pytest

from agentic_cate import download_checkpoint as module


def fake_download(monkeypatch, content):
    monkeypatch.setattr(module, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(content))


def test_download_checks_hash_and_reuses_valid_artifact(tmp_path, monkeypatch):
    content = b"checkpoint bytes, not an executable pickle"
    fake_download(monkeypatch, content)
    destination = tmp_path / "subfolder" / "model.pth"
    assert module.download_checkpoint(destination) == destination
    assert destination.read_bytes() == content
    # An existing valid file must not access the network.
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("unnecessary download"))
    module.download_checkpoint(destination)


def test_invalid_existing_artifact_requires_explicit_overwrite(tmp_path, monkeypatch):
    content = b"valid checkpoint"
    fake_download(monkeypatch, content)
    destination = tmp_path / "model.pth"
    destination.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError, match="--overwrite"):
        module.download_checkpoint(destination)
    assert destination.read_bytes() == b"preserve me"
    module.download_checkpoint(destination, overwrite=True)
    assert destination.read_bytes() == content


def test_bad_download_preserves_destination_and_cleans_temporary(tmp_path, monkeypatch):
    fake_download(monkeypatch, b"expected")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    destination = tmp_path / "model.pth"
    destination.write_bytes(b"original")
    with pytest.raises(ValueError, match="SHA-256"):
        module.download_checkpoint(destination, overwrite=True)
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]
