from typing import Dict


def copy_non_null_dict(original_dict: Dict) -> Dict:
    non_null_dict = original_dict.copy()
    for k, v in original_dict.items():
        if v is None:
            del non_null_dict[k]

    return non_null_dict
