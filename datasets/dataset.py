from __future__ import absolute_import, division, print_function

import tensorflow as tf
from pathlib import *
import random, sys, time

def dataset(data_root, batch_size, verbose=False, autotune=tf.data.experimental.AUTOTUNE):
    all_image_paths, label_names = dir_to_paths_and_labels(data_root)
    unbatched_ds = unbatched_dataset(all_image_paths, label_names, verbose=verbose)

    ds = unbatched_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_paths)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=autotune)
    return ds, label_names, all_image_paths

def all_dataset(data_root, verbose=False):
    all_image_paths, label_names = dir_to_paths_and_labels(data_root)
    unbatched_ds = unbatched_dataset(all_image_paths, label_names, verbose=verbose)
    return unbatched_ds, label_names, all_image_paths

def unbatched_dataset(all_image_paths, label_names, verbose=False):
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[Path(path).parent.name] for path in all_image_paths]

    if verbose:
        print("label to index map ", label_to_index)
        print("First 10 labels indices: ", all_image_labels[:10])
        print("First 10 paths indices: ", all_image_paths[:10])

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    return image_label_ds

def dir_to_paths_and_labels(data_root):
    all_image_paths = dir_to_files(data_root)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    return all_image_paths, label_names

def dir_to_files(data_root):
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths if path.suffix != '']
    random.shuffle(all_image_paths)
    return all_image_paths

#
# If imagenet weights are being loaded, input must have a static square 
#    shape(one of (96, 96), (128, 128), (160, 160),(192, 192), or (224, 224)
def preprocess_image(image, sizes=(96, 3)):
  image = tf.image.decode_jpeg(image, channels=sizes[1])
  image = tf.image.resize_images(image, [sizes[0], sizes[0]])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  print(f"{path}")
  image = tf.read_file(path)
  return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label


#######################################
# import matplotlib.pyplot as plt

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

