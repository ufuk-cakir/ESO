import os

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def plot_chromosome(
    chromosome, image_height, title, results_path=None, name="current_best_chromosome"
):
    plt.figure(figsize=(4.5, 4.5))
    for gene in chromosome.get_genes():
        position = gene.get_band_position()
        height = gene.get_band_height()

        # Create a horizontal span
        plt.axhspan(position, position + height, alpha=0.5)
        plt.ylim(0, image_height)

    plt.gca().invert_yaxis()
    rounded_fitness = round(chromosome.get_fitness(), 4)
    rounded_metric = round(chromosome.get_metric(), 4)
    plt.title("Fitness: " + str(rounded_fitness))
    plt.suptitle(
        title
        + ": "
        + str(rounded_metric)
        + ";Parameters:"
        + str(chromosome.get_trainable_parameters())
    )
    plt.tight_layout()
    if results_path is not None:
        plt.gcf().savefig(os.path.join(results_path, f"{name}.png"))
    return plt.gcf()


def log_tensorboard(
    best_chromosome,
    epoch,
    writer,
    tensorboard_log_dir,
    image_height,
    metric_name,
    results_path=None,
):
    if tensorboard_log_dir is None:
        return

    if metric_name == "f1":
        suptitle_name = "F1-Score"
    else:
        suptitle_name = metric_name.capitalize()

    best_chromosome_fitness = best_chromosome.get_fitness()
    writer.add_scalar("Best Chromosome Fitness", best_chromosome_fitness, epoch)

    writer.add_scalar(
        "Best Chromosome Number of Bands", best_chromosome.num_genes, epoch
    )
    writer.add_scalar(
        f"Best Chromosome {suptitle_name}", best_chromosome.get_metric(), epoch
    )
    writer.add_scalar(
        "Best Chromosome Trainable Parameters",
        best_chromosome.get_trainable_parameters(),
        epoch,
    )
    # Create image
    figure = plot_chromosome(best_chromosome, image_height, suptitle_name, results_path)
    writer.add_figure("Best Chromosome", figure, epoch)
    plt.close()


def setup_tensorboard(tensorboard_log_dir, logger):
    if tensorboard_log_dir is not None:
        tensorboard_log_dir = os.path.join(
            tensorboard_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        logger.debug(f"Logging training to {tensorboard_log_dir}")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_log_dir)
        return writer
    else:
        return None


def setup_logger(logger, log_path, log_level, name=None, add_stream_handler=True):
    if logger is not None:
        return logger
    else:
        import logging

        if name is None:
            name = __name__
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        if add_stream_handler:
            logger.addHandler(logging.StreamHandler())
        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)
            logger.addHandler(
                logging.FileHandler(os.path.join(log_path, f"{name}.log"))
            )
    return logger
