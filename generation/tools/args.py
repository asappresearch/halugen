import argparse
from typing import Any
from tools.logging import LOG_LEVELS, apply_log_level

class ArgParser(argparse.ArgumentParser):
    """
    Subclass of argparse.AgumentParser that will automatically accept `--log-level`
    as an arg and apply the log_level globally for the `logging` library
    """

    def __init__(self, *args: Any, default_log_level: str = "INFO", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--log-level",
            type=str,
            choices=LOG_LEVELS.keys(),
            default=default_log_level,
            help="Set the logging level",
        )

    def parse_args(self, *args: Any, **kwargs: Any) -> Any:
        parsed_args = super().parse_args(*args, **kwargs)
        if parsed_args and parsed_args.log_level:
            apply_log_level(log_level=parsed_args.log_level)
        return parsed_args
