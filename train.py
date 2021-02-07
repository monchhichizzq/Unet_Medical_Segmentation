# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 23:48
# @Author  : Zeqi@@
# @FileName: train.py
# @Software: PyCharm


from Preprocess.Data_Loader import Generator
from models.Unet import Unet
from Loss.metrics import CE, dice_loss_with_CE, Iou_score, f_score

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

if __name__ == "__main__":
    # Hyperparameters
    log_dir = "logs/"
    inputs_size = [512, 512, 3]

    lr = 1e-3
    Init_Epoch = 0
    Freeze_Epoch = 500
    Batch_size = 4
    num_classes = 2
    dice_loss = True

    # Model
    model = Unet(inputs_size, num_classes)
    model.summary()

    # Data
    with open(r"./Medical_Datasets/ImageSets/Segmentation/train.txt", "r") as f:
        train_lines = f.readlines()
        print('Train on {} samples, with batch size {}.'.format(len(train_lines), Batch_size))

    # Callbacks
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-f{_f_score:.3f}-IOU{_Iou_score:.3f}.h5',
                                        monitor='loss', save_weights_only=False, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)



    # Loss
    model.compile(loss=dice_loss_with_CE() if dice_loss else CE(),
                  optimizer=Adam(lr=lr),
                  metrics=[f_score(), Iou_score()])

    gen = Generator(Batch_size, train_lines, inputs_size, num_classes).generate()

    model.fit_generator(gen,
                        steps_per_epoch=max(1, len(train_lines) // Batch_size),
                        epochs=Freeze_Epoch,
                        initial_epoch=Init_Epoch,
                        callbacks=[checkpoint_period, reduce_lr])

