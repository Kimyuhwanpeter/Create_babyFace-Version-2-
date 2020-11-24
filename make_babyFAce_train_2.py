# -*- coding:utf-8 -*-
from absl import flags
from make_babyFace_model_2 import *
from random import random, shuffle

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

flags.DEFINE_string("A_txt_path", "", "A text path")

flags.DEFINE_string("A_img_path", "", "A image path")

flags.DEFINE_string("B_txt_path", "", "B text path")

flags.DEFINE_string("B_img_path", "", "B image path")

flags.DEFINE_string("C_txt_path", "", "C text path")

flags.DEFINE_string("C_img_path", "", "C image path")

flags.DEFINE_integer("load_size", 266, "Original input size")

flags.DEFINE_integer("img_size", 256, "Model input size")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_integer("learning_decay", 100, "Leaning rate decay epochs")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("save_samples", "", "Save training sample images")

flags.DEFINE_string("graphs", "", "Saving loss graphs path")

FLAGS = flags.FLAGS
FLAGS(sys,argv)

len_dataset = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)
len_dataset = len(len_dataset)
G_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.learning_decay * len_dataset)
D_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.learning_decay * len_dataset)
g_optim = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(D_lr_scheduler, beta_1=0.5)

def AB_input_func(A, B):

    A_img = tf.io.read_file(A)
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])

    B_img = tf.io.read_file(B)
    B_img = tf.image.decode_jpeg(B_img, 3)

    return A_img, B_img

def main(): # 로스를 변환해서 쓰자
    from_father_mod = parents_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    from_mother_mod = parents_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    from_baby_mod = baby_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    father_discrim = discrim(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    mother_discrim = discrim(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    baby_discrim = discrim(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    from_father_mod.summary()
    from_mother_mod.summary()
    from_baby_mod.summary()
    father_discrim.summary()
    mother_discrim.summary()
    baby_discrim.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(from_father_mod=from_father_mod,
                                   from_mother_mod=from_mother_mod,
                                   from_baby_mod=from_baby_mod,
                                   g_optim=g_optim,
                                   d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("====================================================================")
            print("Succeed restoring '{}'".format(ckpt_manager.latest_checkpoint))
            print("====================================================================")

    if FLAGS.train:
        count = 0

        A_img = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_img = [FLAGS.A_img_path + img for img in A_img]

        B_img = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_img = [FLAGS.B_img_path + img for img in B_img]

        C_img = np.loadtxt(FLAGS.C_txt_path, dtype="<U100", skiprows=0, usecols=0)
        C_img = [FLAGS.C_img_path + img for img in C_img]

        for epoch in range(FLAGS.epochs):
            np.random.shuffle(A_img)
            np.random.shuffle(B_img)
            np.random.shuffle(C_img)

            AB_gener = tf.data.Dataset.from_tensor_slices((A_img, B_img))
            AB_gener = AB_gener.shuffle(len(A_img))
            AB_gener = AB_gener.map(AB_input_func)


if __name__ == "__main__":
    main()