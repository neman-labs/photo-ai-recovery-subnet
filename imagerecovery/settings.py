import ast
from contextlib import suppress
from os import getenv


def get_env(name: str, default=None, *, required=True, is_list=False):
    value = getenv(name, default)

    if value is None and required:
        raise ValueError(f"Environment variable {name} is not set and has no default value")

    with suppress(Exception):
        value = ast.literal_eval(value)

    if is_list and isinstance(value, str):
        value = value.split(",")

    return value


#  TODO(developer): Add your target environment variables here.
MY_ENV_VAR = get_env("MY_ENV_VAR", "default_value")
