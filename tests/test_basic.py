def test_import():
    try:
        import polars_catchers

        assert polars_catchers.__name__ == "polars_catchers"
    except ImportError:
        assert False, "Failed to import polars_catchers"
