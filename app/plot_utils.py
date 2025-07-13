import os
import matplotlib.pyplot as plt
import seaborn as sns

class PlotUtils:
    @staticmethod
    def plot_loss(loss_history, title='Función de pérdida por época', save_path=None):
        plt.figure()
        plt.plot(loss_history, marker='o')
        plt.title(title)
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.grid(True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, title='Matriz de confusión', save_path=None):
        plt.figure()
        xticks = class_names if class_names is not None else 'auto'
        yticks = class_names if class_names is not None else 'auto'
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticks, yticklabels=yticks)
        plt.title(title)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 