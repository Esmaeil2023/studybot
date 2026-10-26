import pytest

from sample_regulation import build_handbook_pdf, build_pdf
from studybot.ingest import ChunkConfig, DocConfig


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    return build_pdf(tmp_path_factory.mktemp("pdf") / "sample_regulation.pdf")


@pytest.fixture(scope="session")
def handbook_pdf(tmp_path_factory):
    return build_handbook_pdf(tmp_path_factory.mktemp("pdf") / "sample_handbook.pdf")


@pytest.fixture
def doc(sample_pdf):
    return DocConfig(
        doc_id="test-fpo",
        file=str(sample_pdf),
        title="Studien- und Fachprüfungsordnung (Testfassung)",
        short_title="FPO Test",
        version="2024-10-01",
        kind="regulation",
    )


@pytest.fixture
def handbook_doc(handbook_pdf):
    return DocConfig(
        doc_id="test-mhb",
        file=str(handbook_pdf),
        title="Module Handbook (test)",
        short_title="MHB Test",
        version="2026-09-07",
        kind="handbook",
    )


@pytest.fixture
def cfg():
    """The real defaults."""
    return ChunkConfig()


@pytest.fixture
def small_cfg():
    """Small limits, so the short test document has to be split."""
    return ChunkConfig(max_words=80, min_words=20)
