from torch.utils.tensorboard import SummaryWriter

class TBWriter:
    def __init__(self, log_dir='runs/default_experiment'):
        """
        Initializes the TensorBoard writer.

        Args:
            log_dir (str): Directory where TensorBoard logs will be saved.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log_scalar(self, tag, value, epoch):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): Name of the metric.
            value (float): Value of the metric.
            epoch (int): Current epoch number.
        """
        self.writer.add_scalar(tag, value, epoch)
    
    def log_image(self, tag, img_tensor, epoch):
        """
        Logs an image to TensorBoard.

        Args:
            tag (str): Name of the image.
            img_tensor (Tensor): Image tensor.
            epoch (int): Current epoch number.
        """
        self.writer.add_image(tag, img_tensor, epoch)
    
    def log_histogram(self, tag, values, epoch):
        """
        Logs a histogram to TensorBoard.

        Args:
            tag (str): Name of the histogram.
            values (Tensor): Values to histogram.
            epoch (int): Current epoch number.
        """
        self.writer.add_histogram(tag, values, epoch)
    
    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()