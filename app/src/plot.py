import matplotlib.pyplot as plt
import os
import seaborn as sns

class PlotUtils:
    @staticmethod
    def plot_loss(loss_history, title='Loss function by epochs', save_path=None):
        """
        Plot the loss history over epochs.

        Args:
            loss_history (list or array): Sequence of loss values per epoch.
            title (str, optional): Title of the plot. Defaults to 'Loss function by epochs'.
            save_path (str, optional): If provided, saves the plot to this path. Otherwise, displays the plot.
        """
        plt.figure()  # Create a new figure
        plt.plot(loss_history, marker='o')  # Plot loss values with markers
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        if save_path:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_loss_vs_accuracy(loss_history, accuracy_history, title='Loss vs Accuracy by Epoch', save_path=None):
        """
        Plot both loss and accuracy over epochs on the same figure.

        Args:
            loss_history (list or array): Sequence of loss values per epoch.
            accuracy_history (list or array): Sequence of accuracy values per epoch.
            title (str, optional): Title of the plot. Defaults to 'Loss vs Accuracy by Epoch'.
            save_path (str, optional): If provided, saves the plot to this path. Otherwise, displays the plot.
        """
        plt.figure()  # Create a new figure
        plt.plot(loss_history, label='Loss', color='red')  # Plot loss in red
        plt.plot(accuracy_history, label='Accuracy', color='blue')  # Plot accuracy in blue
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_path:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None):
        """
        Plot a confusion matrix using a heatmap.

        Args:
            cm (array-like): Confusion matrix values.
            class_names (list, optional): List of class names for axis labels. If None, uses 'auto'.
            title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
            save_path (str, optional): If provided, saves the plot to this path. Otherwise, displays the plot.
        """
        plt.figure()  # Create a new figure
        xticks = class_names if class_names is not None else 'auto'  # Set x-axis labels
        yticks = class_names if class_names is not None else 'auto'  # Set y-axis labels
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticks, yticklabels=yticks)  # Draw heatmap
        plt.title(title)
        plt.xlabel('Prediction')
        plt.ylabel('Real')
        if save_path:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
