# TORCH.OPTIM
# https://pytorch.org/docs/stable/optim.html#module-torch.optim

# What should I do when my neural network doesn't learn?
# https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn

# https://arxiv.org/abs/2005.14600
# Fixed-size Objects Encoding for Visual Relationship Detection

# Detecting fixed size objects in variable sized images
# https://stackoverflow.com/questions/64861685/detecting-fixed-size-objects-in-variable-sized-images

# How to deal with input size of Faster RCNN and VGG16 /(ㄒoㄒ)/~~?
# https://github.com/jwyang/faster-rcnn.pytorch/issues/484

from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
# from model_mobilenet_v3 import create_model
from model_resnet50 import create_model
from utils import Averager
# from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

import torch
import matplotlib.pyplot as plt
import time

from train_func import *

# name to save the trained model with
MODEL_NAME = 'model'

plt.style.use('ggplot')


# if __name__ == '__main__':
import random
import string

serie = f"{time.strftime('%Y%m%d')[2:]}_{''.join(random.choice(string.ascii_uppercase) for i in range(2))}" # Fecha + Random
print(f"Serie: {serie}")
# initialize the model and move to the computation device
model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# model_path = "/content/drive/MyDrive/PAFAC/obj_detection/outputs/220106_CQ_model_8.pth"
# checkpoint = torch.load(model_path,map_location=torch.device('cuda'))
# model.load_state_dict(checkpoint)

# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer

lr=0.001 # 0.001
momentum=0.9
weight_decay=0.0005

print(f"lr={lr}; momentum={momentum}; weight_decay={weight_decay};")
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  #  step_size=8, gamma=0.1

# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# initialize the Averager class
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# train and validation loss lists to store loss values of all...
# ... iterations till ena and plot graphs for all iterations
train_loss_list = []
val_loss_list = []

# whether to show transformed images from data loader or not
if VISUALIZE_TRANSFORMED_IMAGES:
    from utils import show_tranformed_image
    show_tranformed_image(train_loader)

# start the training epochs
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

    # reset the training and validation loss histories for the current epoch
    train_loss_hist.reset()
    val_loss_hist.reset()

    # create two subplots, one for each, training and validation
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    # start timer and carry out training and validation
    start = time.time()
    train_loss = train(train_loader, model)

    val_loss = validate(valid_loader, model)
    print(f"Epoch #{epoch} train loss:      {train_loss_hist.value:.3f}")   
    print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

    if ((epoch+1) % SAVE_MODEL_EPOCH == 0) or (epoch+1) == NUM_EPOCHS: # save model after every n epochs
        torch.save(model.state_dict(), f"{OUT_DIR}/{serie}_{MODEL_NAME}_{epoch+1}.pth")
        print('SAVING MODEL COMPLETE...\n')
    
    if ((epoch+1) % SAVE_PLOTS_EPOCH == 0) or (epoch+1) == NUM_EPOCHS: # save loss plots after n epochs
        train_ax.plot(train_loss, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        valid_ax.plot(val_loss, color='red')
        valid_ax.set_xlabel('iterations')
        valid_ax.set_ylabel('validation loss')
        figure_1.savefig(f"{OUT_DIR}/{serie}_train_loss_{epoch+1}_{train_loss_hist.value:.3f}.png")
        figure_2.savefig(f"{OUT_DIR}/{serie}_valid_loss_{epoch+1}_{val_loss_hist.value:.3f}.png")
        print('SAVING PLOTS COMPLETE...')
    
    plt.close('all')

    scheduler.step()
    # sleep for 5 seconds after each epoch
    time.sleep(5)