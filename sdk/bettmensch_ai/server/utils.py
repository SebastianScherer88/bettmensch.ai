from typing import Dict


def copy_non_null_dict(original_dict: Dict) -> Dict:
    """Utility function that copies a dictionary without any of its None type
    values (however deep they might be nested).

    Args:
        original_dict (Dict): The dictionary to copy

    Returns:
        Dict: The copied dictionary without any None type values.
    """
    non_null_dict = original_dict.copy()

    for k, v in original_dict.items():
        if v is None:
            del non_null_dict[k]
        elif isinstance(v, dict):
            non_null_dict[k] = copy_non_null_dict(v)
        else:
            pass

    return non_null_dict
