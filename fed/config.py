# General
LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
IMAGE_SIZE = (48, 48)

# Preprocessing
RANDOM_ROTATION = 10
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.35,)

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY = 0.5
LEARNING_RATE_DECAY_STEP = 2
EARLY_STOP_PATIENCE = 7

# Evaluation
RECTANGLE_COLOR = (255, 105, 65)  # BGR
TEXT_COLOR = (255, 255, 255)  # BGR
