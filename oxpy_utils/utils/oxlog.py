import atexit
import logging
import multiprocessing
import multiprocessing.util  # imported before atexit.register below, see close_all_handlers
import os
import sys
import weakref
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

from .sync_queue import SyncQueue

# every OxLogHandler not yet closed, so they can all be closed at exit
open_handlers = weakref.WeakSet()


class OxLogHandler:
    def __init__(self, name: str, verbose: bool = True, log_to_dir: Path = Path(".")):
        """
        Initialize the OxLogHandler instance.

        Records sent to its loggers (spinoff(), also from forked worker processes) go through a multiprocessing
        queue to a listener thread in this process, which writes them out. close() stops that thread after it
        has written every queued record; it is called when the handler is garbage collected, and for every open
        handler at interpreter exit (close_all_handlers).

        Args:
            name (str): Name for the log file and logger.
            verbose (bool): If True, logs are also printed to the console.
        """
        # SyncQueue, not multiprocessing.Queue: a worker whose main thread holds the GIL in a long C call
        # (oxpy running a simulation) must not be able to hold the queue's shared write lock (see sync_queue.py)
        self.log_queue = SyncQueue()
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
        # forked worker processes get copies of this object; only the creating process owns the listener
        self.owner_pid = os.getpid()
        self.closed = False
        open_handlers.add(self)

    def set_verbose(self, bIsVerbose: bool=True):
        for handler in self.log_listener.handlers:
            handler.setLevel(logging.INFO if bIsVerbose else logging.WARNING)

    def close(self):
        """
        Stop the listener thread once it has written every record already queued, and close the log file.
        Does nothing if already closed, or when called in a forked copy (a child must not stop the parent's
        listener, which would happen as the stop sentinel goes through the shared queue).
        """
        if getattr(self, "closed", True) or os.getpid() != self.owner_pid:
            return
        self.closed = True
        open_handlers.discard(self)
        self.log_listener.stop()
        for handler in self.log_listener.handlers:
            handler.close()

    def __del__(self):
        """
        Destructor to stop the log listener.
        """
        try:
            self.close()
        except (RuntimeError, OSError, ValueError):
            # garbage collected during interpreter shutdown, after close_all_handlers: the queue's pipe or lock
            # may already be gone, so stop() cannot send its sentinel
            pass

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


def close_all_handlers():
    """
    Close every open OxLogHandler. Registered with atexit: without it, listener threads are still blocked on their
    queues when the interpreter finalizes, the queues are torn down under them, and the threads' errors at
    shutdown can abort the interpreter (and records still queued are never written). atexit runs handlers in
    reverse order of registration, so this runs before multiprocessing's own exit handler (registered when
    multiprocessing.util is imported above) finalizes the queues.
    """
    for handler in list(open_handlers):
        handler.close()


atexit.register(close_all_handlers)
