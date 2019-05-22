from __future__ import absolute_import, division, print_function

import tensorflow as tf
from pathlib import *
import time, click
from datasets import dataset
from networks import mobile_net

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

print(f'TF v {tf.VERSION}')
print(f'autotune: {AUTOTUNE}')

@click.command()
@click.argument('dataset', type=click.Path(exists=True), default="./data")
@click.option('--n-epochs', type=int, default=5)
@click.option('--lr', type=float, default=3e-4)
@click.option('--batch-size', type=int, default=32)
@click.option('--verbose', is_flag=True, default=False)
def main(n_epochs, lr, batch_size, dataset, verbose):
    train(n_epochs, lr, batch_size, Path(dataset), verbose)

def train(n_epochs, learning_rate, batch_size, data_root, verbose):
    ds, label_names, all_image_paths = dataset(data_root, batch_size, verbose=verbose)

    # print(f"{ds}, {label_names}, {all_image_paths}")
    # mobile_net.trainable=False

    keras_ds = ds.map(change_range)
    print(f" {keras_ds}")
    image_batch, label_batch = next(iter(keras_ds))

    model = mobile_net(label_names, pic_size=96)

    # mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)

    # # if verbose:
    # #     feature_map_batch = mobile_net(image_batch)
    # #     print(feature_map_batch.shape)

    # model = tf.keras.Sequential([
    #     mobile_net,
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(len(label_names))])

    logit_batch = model(image_batch).numpy()
    if verbose:
        print(f"min logit: {logit_batch.min()} ")
        print(f"max logit: {logit_batch.max()} ")
        print()
        print(f"Shape: {logit_batch.shape}")

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

    if verbose:
        print(f"Trainable variables: {len(model.trainable_variables)}")
        model.summary()

    steps_per_epoch=tf.ceil(len(all_image_paths)/batch_size).numpy()
    model.fit(ds, epochs=n_epochs, steps_per_epoch=int(steps_per_epoch))

    def timeit(ds, batches=2*steps_per_epoch+1):
        overall_start = time.time()
        # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
        # before starting the timer
        it = iter(ds.take(batches+1))
        next(it)

        start = time.time()
        for i,(images,labels) in enumerate(it):
            if i%10 == 0:
              print('.',end='')
        print()
        end = time.time()

        duration = end-start
        print("{} batches: {} s".format(batches, duration))
        print("{:0.5f} Images/s".format(batch_size*batches/duration))
        print("Total time: {}s".format(end-overall_start))

    if verbose:
        timeit(ds)


def change_range(image,label):
  return 2*image-1, label

if __name__ == '__main__':
    main()  