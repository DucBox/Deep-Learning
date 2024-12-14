import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from paths import TRAIN_CSV, VALID_CSV, TRAIN_DIR, VALID_DIR


def load_data(target_size=(224, 224), batch_size=64):
    """
    Load training and validation datasets using ImageDataGenerator.

    Args:
        target_size (tuple): Image size for resizing.
        batch_size (int): Batch size for the data generators.

    Returns:
        tuple: Training and validation generators.
    """
    # Load CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)

    # Data augmentation
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col=train_df.columns[1:],  # Multi-label columns
        class_mode="raw",
        target_size=target_size,
        batch_size=batch_size,
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=VALID_DIR,
        x_col="filename",
        y_col=valid_df.columns[1:],  # Multi-label columns
        class_mode="raw",
        target_size=target_size,
        batch_size=batch_size,
    )

    return train_generator, valid_generator
