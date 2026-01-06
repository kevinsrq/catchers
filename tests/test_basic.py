def test_import():
    try:
        import catchers

        assert catchers.__name__ == "catchers"
    except ImportError:
        assert False, "Failed to import catchers"
