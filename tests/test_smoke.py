import authorlens


def test_version():
    assert authorlens.__version__ == "0.1.0"


def test_stub_imports():
    from authorlens import features, embedder, scorer, cli, api  # noqa: F401
