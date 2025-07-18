"""This module provides general useful functions"""

import yaml


def load_config(path: str = "scripts/config.yaml") -> dict:
    """
    Load configuration file.

    :param path: Path to configuration file.
    :return: Configuration dictionary.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def normalise_scalar(min_val: float, max_val: float, val: float) -> float:
    """
    Normalise a given scalar value.

    :param min_val: Minimum scalar value.
    :param max_val: Maximum scalar value.
    :param val: Scalar value to normalise.

    :return: Normalised scalar value.
    """
    return (val - min_val) / (max_val - min_val)


def normalise_dict(dict_to_normalise: dict) -> dict:
    """
    Normalise the values in the dictionary.

    :param dict_to_normalise: Dictionary for which the values will be normalised.
    :return: Normalised dictionary.
    """
    min_v, max_v = min(dict_to_normalise.values()), max(dict_to_normalise.values())
    return {
        k: (v - min_v) / (max_v - min_v) if max_v != min_v else 0
        for k, v in dict_to_normalise.items()
    }
