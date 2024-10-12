import json
import os

from utils.logger import Logger
from utils.tb_writer import TBWriter
from utils.metrics_logger import MetricsLogger

class MetricsHandler:
    def __init__(self, experiment_dir, log_dir):
        """
        Initializes the MetricsHandler with necessary utilities.

        Args:
            experiment_dir (str): Directory where models and metrics will be saved.
            log_dir (str): Directory for TensorBoard logs.
        """
        self.logger = Logger("rich")
        self.tb_writer = TBWriter(log_dir=log_dir)
        self.metrics_logger = MetricsLogger(experiment_dir=experiment_dir)
    
    def handle_metrics(self, metrics_dict, epoch):
        """
        Handles logging of metrics by saving to JSON, TensorBoard, and displaying in console.

        Args:
            metrics_dict (dict): Dictionary containing metric names and their values.
            epoch (int): Current epoch number.
        """
        # Log metrics to JSON
        self.metrics_logger.log_metrics(epoch=epoch, metrics_dict=metrics_dict)
    
        # Log metrics to TensorBoard
        for metric, value in metrics_dict.items():
            tag = f"Validation/{metric.replace(' ', '_')}"
            self.tb_writer.log_scalar(tag, value, epoch)
    
        # Display metrics in console
        table_title = f"Epoch {epoch} Summary"
        self.logger.print_table(table_title, metrics_dict)
    
    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.tb_writer.close()