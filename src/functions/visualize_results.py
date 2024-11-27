import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os


def visualize_results(history, labels, test_gen, test_data, model, results_path):

    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
        
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.set_theme(style='whitegrid', rc={'axes.facecolor':'#5fa1bc'})

    ax[0].plot(history.history['accuracy'], color='blue', label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], color='red', label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'], color='blue', label='Train Loss')
    ax[1].plot(history.history['val_loss'], color='red', label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    accuracy_loss_path = os.path.join(results_path, 'accuracy_loss_plot.png')
    plt.savefig(accuracy_loss_path)
    plt.close(fig)



    pred = model.predict(test_gen)
    pred = np.argmax(pred, axis=1)
    pred = [labels[i] for i in pred]

    cm = confusion_matrix(test_data['Labels'], pred)
    class_report = classification_report(test_data['Labels'], pred)

    class_report_path = os.path.join(results_path, 'classification_report.txt')
    with open(class_report_path, 'w') as file:
        file.write(class_report)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels.values(), yticklabels=labels.values())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    confusion_matrix_path = os.path.join(results_path, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close(fig)

    fig, axes = plt.subplots(4,3, figsize=(12,8), subplot_kw={'xticks':[], 'yticks':[]})

    for i, axes in enumerate(axes.flat):
        axes.imshow(plt.imread(test_data.Images.iloc[i+1]))
        axes.set_title(f'True: {test_data.Labels.iloc[i+1]}\n Predicted: {pred[i+1]}')

    plt.tight_layout()
    predictions_visualization_path = os.path.join(results_path, 'predictions_visualization.png')
    plt.savefig(predictions_visualization_path)
    plt.close(fig)
