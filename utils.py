# Importa libs úteis para avaliação dos modelos
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def show_predict_infos(y, predict, title="", cmap="Blues"):
    accuracy = accuracy_score(y, predict)
    accuracy_percent = accuracy * 100
    print(f"A acurácia no conjunto de testes: {accuracy_percent:.2f}%")

    f1 = f1_score(y, predict, average="macro")
    f1_percent = f1 * 100
    print(f"A F1 no conjunto de testes: {f1_percent:.2f}%")

    recall = recall_score(y, predict, average="macro")
    recall_percent = recall * 100
    print(f"A recall no conjunto de testes: {recall_percent:.2f}%")

    precision = precision_score(y, predict, average="macro")
    precision_percent = precision * 100
    print(f"A precision no conjunto de testes: {precision_percent:.2f}%")

    # Mostra um relatório com as métricas de classificação por classe e as métricas calculadas sobre o conjunto todo.
    print(classification_report(y, predict))

    ConfusionMatrixDisplay.from_predictions(y, predict, colorbar=False, cmap=cmap)
    if len(title) > 0:
        plt.title(f"Matriz de Confusão {title}")
    plt.xlabel("Rótulo Previsto")
    plt.ylabel("Rótulo Real")
    plt.show()

    return accuracy, f1, recall, precision
    
