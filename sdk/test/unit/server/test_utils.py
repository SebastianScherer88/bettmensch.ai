from bettmensch_ai.server.utils import copy_non_null_dict


def test_copy_non_null_dict():

    test_dict = {
        "a": None,
        "b": 2,
        "c": {"d": None, "e": 5, "f": {"g": None, "h": 8}},
    }

    non_null_dict = copy_non_null_dict(test_dict)

    assert non_null_dict == {"b": 2, "c": {"e": 5, "f": {"h": 8}}}
