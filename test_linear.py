import tensorflow as tf
import tensorflow.feature_column as fc 
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import random, functools # os,sys, 

tf.enable_eager_execution()

def all_dataset(data_root, verbose=False):
    all_image_paths, label_names = dir_to_paths_and_labels(data_root)
    unbatched_ds = unbatched_dataset(all_image_paths, label_names, verbose=verbose)
    return unbatched_ds, label_names, all_image_paths

def unbatched_dataset(all_image_paths, label_names, verbose=False):
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    # def limited_dataset(all_paths, threshold = 16):
    #     counter = defaultdict(int)        
    #     out = []
    #     for path in all_paths:
    #       if counter[Path(path).parent.name] < threshold:
    #         out.append(path)
    #         counter[Path(path).parent.name]+=1
    #     return out 

    # all_image_paths = limited_dataset(all_image_paths)

    all_image_labels = [label_to_index[Path(path).parent.name] for path in all_image_paths]

    if verbose:
        print("label to index map ", label_to_index)
        print("First 32 labels indices: ", all_image_labels[:32])
        print("First 32 paths indices: ", all_image_paths[:32])

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
  # return dict({'image': load_and_preprocess_image(path)}), label
  val = load_and_preprocess_image(path)
  print(f"load_and_preprocess_from_path_label: {val}")
  return dict({'image': val}), label

def custom_input_fn(train_root, num_epochs=5, batch_size=64):
    ds, label_names, all_image_paths = all_dataset(train_root, verbose=True)
    ds = ds.batch(batch_size).repeat(num_epochs) # ds.repeat(num_epochs) # 
    # for feature_batch, label_batch in ds.take(1):
    #   print('Some feature keys:', list(feature_batch)[:5])
    #   print()
    #   print('A batch of Labels:', label_batch )
    return ds

def main():
    classifier = tf.estimator.LinearClassifier(feature_columns=[fc.numeric_column(key='image',shape=(96,96,3))])
    train_inpf = functools.partial(custom_input_fn, Path("data"))
    classifier.train(train_inpf)

if __name__ == "__main__":
    main()


