import sys
import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path


class OxLogHandler:
    def __init__(self, name: str, verbose: bool = True, log_to_dir: Path = Path(".")):
        """
        Initialize the OxLogHandler instance.

        Args:
            name (str): Name for the log file and logger.
            verbose (bool): If True, logs are also printed to the console.
        """
        self.log_queue = multiprocessing.Queue()
        self.formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(name)s) %(message)s')

        # Create file handler
        file_handler = logging.FileHandler(log_to_dir/f"{name}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self.formatter)
        
        # Create optional stream handler if verbose
        self.verbose = verbose
        handlers = [file_handler]
        if self.verbose:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(self.formatter)
            handlers.append(stream_handler)

        # Initialize the QueueListener with handlers
        self.log_listener = QueueListener(self.log_queue, *handlers, respect_handler_level=True)
        self.log_listener.start()

    def set_verbose(self, bIsVerbose: bool=True):
        for handler in self.log_listener.handlers:
            handler.setLevel(logging.INFO if bIsVerbose else logging.WARNING)

    def __del__(self):
        """
        Destructor to stop the log listener.
        """
        if hasattr(self, 'log_listener') and self.log_listener:
            self.log_listener.stop()

    def spinoff(self, name: str) -> logging.Logger:
        """
        Create and return a logger configured to use the queue.

        :param: Name of the logger.
        :returns: Configured logger instance.
        """
        logger = logging.getLogger(name)
        # deal with redundant logger names because apparently there's no easy way to un-register a logger
        if len(logger.handlers) < 1:
            logger.setLevel(logging.INFO)

            # Attach a QueueHandler to send log messages to the queue
            queue_handler = QueueHandler(self.log_queue)
            logger.addHandler(queue_handler)
        else:
            # overwrite existing handler
            logger.handlers[0] = QueueHandler(self.log_queue)
        return logger
