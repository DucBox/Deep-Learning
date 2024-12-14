from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import config
from paths import TRAIN_CSV, VALID_CSV, MODEL_SAVE_PATH

print("Training CSV Path:", TRAIN_CSV)
print("Model Save Path:", MODEL_SAVE_PATH)

def build_resnet_model(input_shape=(224, 224, 3), num_classes=len(config.LABELS)):
    """
    Build the ResNet50 model for multi-label classification.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes (multi-label).

    Returns:
        Model: Compiled ResNet50 model.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks():
    """
    Define callbacks for training.

    Returns:
        list: List of Keras callbacks.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
    )

    # Save the best model based on validation loss
    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH, 
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # Log training metrics to a CSV file
    csv_logger = CSVLogger("training_log.csv", append=True)

    return [early_stopping, model_checkpoint, csv_logger]


def train_model(
    model,
    train_generator,
    valid_generator,
    epochs=config.EPOCHS,
    callbacks=None,
    verbose=1,
):
    """
    Train the ResNet50 model.

    Args:
        model (Model): Compiled ResNet50 model.
        train_generator: Training data generator.
        valid_generator: Validation data generator.
        epochs (int): Number of epochs.
        callbacks (list): List of callbacks.
        verbose (int): Verbosity level.

    Returns:
        History: Training history.
    """
    if callbacks is None:
        callbacks = get_callbacks()

    print("Starting model training...")
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    print("Model training completed.")
    return history


if __name__ == "__main__":
    from data.load_data import load_data

    # Load data generators
    train_generator, valid_generator = load_data()

    # Build the model
    model = build_resnet_model()

    # Train the model
    history = train_model(
        model,
        train_generator,
        valid_generator,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(),
    )
