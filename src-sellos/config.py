import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 200 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '../selllos_detection/train/'
# validation images and XML files directory
VALID_DIR = '../selllos_detection/test/'

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'colada', 'junta', 'protuberancia'
]
NUM_CLASSES = 4

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs

print("config.py")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"RESIZE_TO: {RESIZE_TO}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")

print(f"DEVICE: {DEVICE}")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"VALID_DIR: {VALID_DIR}")

print(f"CLASSES: {CLASSES}")
print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"VISUALIZE_TRANSFORMED_IMAGES: {VISUALIZE_TRANSFORMED_IMAGES}")

print(f"OUT_DIR: {OUT_DIR}")
print(f"SAVE_PLOTS_EPOCH: {SAVE_PLOTS_EPOCH}")
print(f"SAVE_MODEL_EPOCH: {SAVE_MODEL_EPOCH}")
