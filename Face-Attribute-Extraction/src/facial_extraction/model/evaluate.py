from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from paths import MODEL_SAVE_PATH


def evaluate_model(model, generator):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (Model): Trained ResNet50 model.
        generator: Validation data generator.

    Returns:
        dict: Evaluation metrics (e.g., precision, recall, F1 score).
    """
    print("Starting evaluation...")
    predictions = model.predict(generator)  # Predict on the validation set
    y_true = generator.labels  # Ground truth labels from the generator

    # Calculate evaluation metrics (e.g., precision, recall, F1 score)
    report = classification_report(y_true, predictions > 0.5, output_dict=True)
    print("Evaluation completed.")
    return report


def load_trained_model(model_path=MODEL_SAVE_PATH):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Model: Loaded Keras model.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")
    return model


if __name__ == "__main__":
    from data.load_data import load_data

    # Load validation data
    _, valid_generator = load_data()

    # Load the trained model
    model = load_trained_model()

    # Evaluate the model
    report = evaluate_model(model, valid_generator)
    print(report)
