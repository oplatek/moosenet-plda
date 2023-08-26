def test_version_available():
    """You need to run pip install to create the moosenet.version module

    For development you just need to install moosenet

    pip install -e .[dev]
    """
    import moosenet.version

    print(moosenet.version.__version__)
