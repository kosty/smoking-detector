from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import *
import random, sys, time, click

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
    ds, label_names, all_image_paths = dataset(data_root, batch_size, verbose)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    # mobile_net.trainable=False

    keras_ds = ds.map(change_range)
    image_batch, label_batch = next(iter(keras_ds))
    feature_map_batch = mobile_net(image_batch)

    if verbose:
        print(feature_map_batch.shape)

    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(label_names))])

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
        print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
        print("Total time: {}s".format(end-overall_start))

    if verbose:
        timeit(ds)



def dataset(data_root, batch_size, verbose):
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[Path(path).parent.name] for path in all_image_paths]

    if verbose:
        print("label to index map ", label_to_index)
        print("First 10 labels indices: ", all_image_labels[:10])
        print("First 10 paths indices: ", all_image_paths[:10])

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    BATCH_SIZE = batch_size
    ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds, label_names, all_image_paths

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def change_range(image,label):
  return 2*image-1, label

if __name__ == '__main__':
    main()  



# ds = image_label_ds.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)



#######################################

# print(f'Image count: {image_count}')


# print(f'Label names: {label_names}')


# print(f'Label index: {label_to_index}')


# print("First 10 labels indices: ", all_image_labels[:10])

# img_path = all_image_paths[0]

# img_raw = tf.read_file(img_path)
# print(repr(img_raw)[:100]+"...")

# img_tensor = tf.image.decode_image(img_raw)

# print(img_tensor.shape)
# print(img_tensor.dtype)

# img_final = tf.image.resize_images(img_tensor, [224, 224])
# img_final = img_final/255.0

# print(img_final.shape)
# print(img_final.numpy().min())
# print(img_final.numpy().max())



# image_path = all_image_paths[0]
# label = all_image_labels[0]

# plt.imshow(load_and_preprocess_image(img_path))
# plt.grid(False)
# plt.xlabel(f'caption_image {img_path}')
# plt.title(label_names[label].title())
# print()

# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# print('shape: ', repr(path_ds.output_shapes))
# print('type: ', path_ds.output_types)
# print()
# print(path_ds)

# for label in label_ds.take(10):
#   print(label_names[label.numpy()])


# print('image shape: ', image_label_ds.output_shapes[0])
# print('label shape: ', image_label_ds.output_shapes[1])
# print('types: ', image_label_ds.output_types)
# print()
# print(image_label_ds)


# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
# ds = image_label_ds.shuffle(buffer_size=image_count)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.

# print("min logit:", logit_batch.min())
# print("max logit:", logit_batch.max())
# print()

# print("Shape:", logit_batch.shape)

