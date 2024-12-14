from paths import TRAIN_CSV, VALID_CSV, TRAIN_DIR, VALID_DIR, MODEL_SAVE_PATH

# Training parameters
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
EPOCHS = 100

# Labels (Update if labels differ in your dataset)
LABELS = ["beard", "earrings", "female", "glass", "hat", "male"]
