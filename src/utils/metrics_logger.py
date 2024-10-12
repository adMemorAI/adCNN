import json
import os

class MetricsLogger:
    def __init__(self, experiment_dir, metrics_file='metrics.json'):
        """
        Initializes the MetricsLogger.

        Args:
            experiment_dir (str): Directory where the metrics file will be saved.
            metrics_file (str): Name of the JSON file to store metrics.
        """
        self.metrics_file = os.path.join(experiment_dir, metrics_file)
        # Initialize the JSON file with an empty list if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump([], f, indent=4)

    def log_metrics(self, epoch, metrics_dict):
        """
        Logs the metrics for a specific epoch.

        Args:
            epoch (int): The current epoch number.
            metrics_dict (dict): Dictionary containing metric names and their values.
        """
        epoch_metrics = {"epoch": epoch}
        epoch_metrics.update(metrics_dict)

        # Read existing metrics
        with open(self.metrics_file, 'r') as f:
            data = json.load(f)

        # Append new metrics
        data.append(epoch_metrics)

        # Write back to the JSON file
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=4)

    def get_all_metrics(self):
        """
        Retrieves all logged metrics.

        Returns:
            list: A list of dictionaries containing metrics for each epoch.
        """
        with open(self.metrics_file, 'r') as f:
            data = json.load(f)
        return data