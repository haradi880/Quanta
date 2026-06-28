from scripts.check_tracked_secrets import scan


def test_tracked_production_files_contain_no_recognized_secrets():
    assert scan() == []
