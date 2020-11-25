# -*- coding:utf-8 -*-
from absl import flags
from make_babyFace_model_2 import *
from random import random, shuffle

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

flags.DEFINE_string("A_txt_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[1]Father/labels.txt", "Training A text path")

flags.DEFINE_string("A_img_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[1]Father/AFAD/", "Training A image path")

flags.DEFINE_string("B_txt_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[2]Mother/labels.txt", "Training B text path")

flags.DEFINE_string("B_img_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[2]Mother/AFAD/", "Training B image path")

flags.DEFINE_string("C_txt_path", "D:/[1]DB/[4]etc_experiment/UTK_face/baby_labels.txt", "Training C text path")

flags.DEFINE_string("C_img_path", "D:/[1]DB/[4]etc_experiment/UTK_face/baby/", "Training C image path")

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

flags.DEFINE_string("save_samples", "C:/Users/Yuhwan/Pictures/sample", "Save training sample images")

flags.DEFINE_string("graphs", "", "Saving loss graphs path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

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
    B_img = tf.image.resize(B_img, [FLAGS.load_size, FLAGS.load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3])

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)
    A_img = A_img / 127.5 - 1.
    B_img = B_img / 127.5 - 1.

    return A_img, B_img

def C_input_func(C):

    C_img = tf.io.read_file(C)
    C_img = tf.image.decode_jpeg(C_img, 3)
    C_img = tf.image.resize(C_img, [FLAGS.load_size, FLAGS.load_size])
    C_img = tf.image.random_crop(C_img, [FLAGS.img_size, FLAGS.img_size, 3])

    if random() > 0.5:
        C_img = tf.image.flip_left_right(C_img)

    C_img = C_img / 127.5 - 1.

    return C_img

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(from_father_mod, from_mother_mod, from_baby_mod,
            father_discrim, mother_discrim, baby_discrim,
            A_batch_img, B_batch_img, C_batch_img):

    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape() as d_tape:
        fake_baby_father = run_model(from_father_mod, A_batch_img, True)
        fake_baby_mother = run_model(from_mother_mod, B_batch_img, True)
        
        #similarity_father = (fake_baby_father * tf.expand_dims(C_batch_img[0], 0)) / (tf.math.sqrt(fake_baby_father*fake_baby_father) * tf.math.sqrt(tf.expand_dims(C_batch_img[0], 0)*tf.expand_dims(C_batch_img[0], 0)))
        #similarity_mother = (fake_baby_mother * tf.expand_dims(C_batch_img[1], 0)) / (tf.math.sqrt(fake_baby_mother*fake_baby_mother) * tf.math.sqrt(tf.expand_dims(C_batch_img[0], 0)*tf.expand_dims(C_batch_img[0], 0)))
        similarity_father = -tf.nn.tanh(tf.abs(tf.expand_dims(C_batch_img[0], 0) - fake_baby_father))
        similarity_mother = -tf.nn.tanh(tf.abs(tf.expand_dims(C_batch_img[1], 0) - fake_baby_mother))
        similarity = tf.concat([similarity_father, similarity_mother], 0)
        baby_part = run_model(from_baby_mod, similarity, True)

        fake_baby_from_father = run_model(father_discrim, similarity_father, True)
        fake_baby_from_mother = run_model(mother_discrim, similarity_mother, True)
        real_baby_from_father = run_model(father_discrim, A_batch_img, True)
        real_baby_from_mother = run_model(mother_discrim, B_batch_img, True)
        fake_baby = run_model(baby_discrim, baby_part, True)
        real_baby = run_model(baby_discrim, C_batch_img, True)

        g_father_Idloss = tf.reduce_mean(tf.abs(similarity_father - tf.expand_dims(C_batch_img[0], 0)))
        g_mother_Idloss = tf.reduce_mean(tf.abs(similarity_mother - tf.expand_dims(C_batch_img[1], 0)))
        g_baby_Idloss = tf.reduce_mean(tf.abs(baby_part - C_batch_img))

        g_loss = tf.reduce_mean((tf.ones_like(fake_baby_from_father) - fake_baby_from_father)**2) \
                + tf.reduce_mean((tf.ones_like(fake_baby_from_mother) - fake_baby_from_mother)**2) \
                + tf.reduce_mean((tf.ones_like(fake_baby) - fake_baby)**2) \
                + (g_baby_Idloss * 10.0) + (g_father_Idloss + g_mother_Idloss) * 5.0

        d_loss = (tf.reduce_mean((tf.zeros_like(fake_baby_from_father) - fake_baby_from_father)**2) + tf.reduce_mean((tf.ones_like(real_baby_from_father) - real_baby_from_father)**2)) * 0.5 \
                + (tf.reduce_mean((tf.zeros_like(fake_baby_from_mother) - fake_baby_from_mother)**2) + tf.reduce_mean((tf.ones_like(real_baby_from_mother) - real_baby_from_mother)**2)) * 0.5 \
                + (tf.reduce_mean((tf.zeros_like(fake_baby) - fake_baby)**2) + tf.reduce_mean((tf.ones_like(real_baby) - real_baby)**2)) * 0.5
    g_grads = g_tape.gradient(g_loss, from_father_mod.trainable_variables + from_mother_mod.trainable_variables)
    g_grads2 = g_tape.gradient(g_loss, from_baby_mod.trainable_variables)

    d_grads = d_tape.gradient(d_loss, father_discrim.trainable_variables + mother_discrim.trainable_variables + baby_discrim.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, from_father_mod.trainable_variables + from_mother_mod.trainable_variables))
    g_optim.apply_gradients(zip(g_grads2, from_baby_mod.trainable_variables))

    d_optim.apply_gradients(zip(d_grads, father_discrim.trainable_variables + mother_discrim.trainable_variables + baby_discrim.trainable_variables))

    return g_loss, d_loss

def save_fig_baby(C, count, name):

    C_real_img = np.zeros([256, 512, 3], dtype=np.float32)
    for i in range(FLAGS.batch_size + 1):

        C_real_img[:, 256*i:256*(i + 1), :] = C[i]

    plt.imsave(FLAGS.save_samples + "/" + name + "_{}.jpg".format(count), C_real_img * 0.5 + 0.5)

def main():
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
            AB_gener = AB_gener.batch(FLAGS.batch_size)
            AB_gener = AB_gener.prefetch(tf.data.experimental.AUTOTUNE)

            C_gener = tf.data.Dataset.from_tensor_slices(C_img)
            C_gener = C_gener.shuffle(len(C_img))
            C_gener = C_gener.map(C_input_func)
            C_gener = C_gener.batch(FLAGS.batch_size + 1)
            C_gener = C_gener.prefetch(tf.data.experimental.AUTOTUNE)

            AB_iter = iter(AB_gener)
            AB_idx = len(A_img) // FLAGS.batch_size
            for step in range(AB_idx):
                C_iter = iter(C_gener)
                C_batch_img = next(C_iter)
                A_batch_img, B_batch_img = next(AB_iter)

                g_loss, d_loss = cal_loss(from_father_mod, from_mother_mod, from_baby_mod,
                                          father_discrim, mother_discrim, baby_discrim,
                                          A_batch_img, B_batch_img, C_batch_img)

                print("Epoch: {} [{}/{}] g_loss = {}, d_loss = {}".format(epoch, step + 1, AB_idx, g_loss, d_loss))

                if count % 100 == 0:
                    father_part = run_model(from_father_mod, A_batch_img, False)
                    mother_part = run_model(from_mother_mod, B_batch_img, False)
                    similarity_father = -tf.nn.tanh(tf.abs(tf.expand_dims(C_batch_img[0], 0) - father_part))
                    similarity_mother = -tf.nn.tanh(tf.abs(tf.expand_dims(C_batch_img[1], 0) - mother_part))
                    similarity = tf.concat([similarity_father, similarity_mother], 0)
                    baby_part = run_model(from_baby_mod, similarity, True)

                    save_fig_baby(C_batch_img, count, "real_C")
                    save_fig_baby(baby_part, count, "fake_C")
                    plt.imsave(FLAGS.save_samples + "/"+ "_real_A_{}.jpg".format(count), A_batch_img[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_samples + "/"+ "_real_B_{}.jpg".format(count), B_batch_img[0] * 0.5 + 0.5)


                count += 1


if __name__ == "__main__":
    main()
