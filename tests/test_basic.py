
def test_import():
    try:
        import catchers
    except ImportError:
        assert False, "Failed to import catchers"
