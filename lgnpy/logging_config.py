import os
import logging
import logging.config


class Logger:
    """
    Common logger for all modules

    Usage:
        from logging_config import CommonLogger
        _logger = CommonLogger(__name__,debug=True).setup_logger()
        _logger.info("Hello logger")

    Sample Output:
        INFO __main__ at 19-Dec-19,13:35:09: Hello logger
    """

    def __init__(self, module_name=None):
        self.env = os.environ.get("ENVIRONMENT", "")
        self.mod_name = module_name if module_name else __name__
        self.logger = None
        self.log_formatter = None
        self.log_handler = logging.StreamHandler()
        self.log_format = logging.Formatter("%(message)s", datefmt="%d-%b-%y,%H:%M:%S")

    def setup_logger(self, debug=False):
        self.debug = debug
        self.logger = logging.getLogger(self.mod_name)
        self.log_handler.setFormatter(self.log_format)
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        return self.logger
