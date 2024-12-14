import json
from sklearn.metrics import classification_report


def evaluate_pipeline(predictions, ground_truth):
    """
    Evaluate the pipeline's predictions against the ground truth.

    Args:
        predictions (list): List of prediction dictionaries.
        ground_truth (list): List of ground truth dictionaries.

    Returns:
        dict: Classification report with precision, recall, and F1 score.
    """
    y_true = []
    y_pred = []

    for gt, pred in zip(ground_truth, predictions):
        y_true.append(gt["labels"])
        y_pred.append(list(pred["classifications"].keys()))

    # Flatten lists and calculate the classification report
    flattened_y_true = [label for sublist in y_true for label in sublist]
    flattened_y_pred = [label for sublist in y_pred for label in sublist]

    return classification_report(flattened_y_true, flattened_y_pred, output_dict=True)


def load_ground_truth(file_path):
    """
    Load ground truth data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing ground truth.

    Returns:
        list: List of dictionaries with ground truth data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    """
    Example usage of pipeline evaluation.
    """
    # Paths to files (to be updated)
    predictions_path = "predictions.json"
    ground_truth_path = "ground_truth.json"

    # Load predictions and ground truth
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    ground_truth = load_ground_truth(ground_truth_path)

    # Evaluate the pipeline
    report = evaluate_pipeline(predictions, ground_truth)
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
