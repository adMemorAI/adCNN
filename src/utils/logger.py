import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

class Logger:
    def __init__(self, name="rich"):
        """
        Initializes the Logger with Rich formatting.

        Args:
            name (str): Name of the logger.
        """
        self.console = Console()
        logging.basicConfig(
            level=logging.INFO,  # Change to DEBUG for more detailed logs
            format='[%(asctime)s] %(message)s',
            datefmt='[%X]',
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(name)
    
    def info(self, message):
        """
        Logs an info level message.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)
    
    def debug(self, message):
        """
        Logs a debug level message.

        Args:
            message (str): The message to log.
        """
        self.logger.debug(message)
    
    def error(self, message):
        """
        Logs an error level message.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)
    
    def print_table(self, title, metrics):
        """
        Displays a formatted table of metrics.

        Args:
            title (str): Title of the table.
            metrics (dict): Dictionary containing metric names and their values.
        """
        table = Table(title=title)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        for metric, value in metrics.items():
            table.add_row(metric, f"{value}")
        
        self.console.print(table)