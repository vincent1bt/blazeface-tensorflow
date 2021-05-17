import numpy as np
import tensorflow as tf

from utils import list_of_boxes

images_paths = np.load('data_files/images_paths.npy')
boxes_dict = np.load('data_files/boxes_dict.npy', allow_pickle=True)

def compute_iou(box, anchor_box):  
  x_min = np.maximum(box[0], anchor_box[0])
  y_min = np.maximum(box[1], anchor_box[1])
  x_max = np.minimum(box[2], anchor_box[2])
  y_max = np.minimum(box[3], anchor_box[3])
  
  overlap_area = np.maximum(0.0, x_max - x_min + 1) * np.maximum(0.0, y_max - y_min + 1)
  
  true_boxes_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
  anchor_boxes_area = (anchor_box[2] - anchor_box[0] + 1) * (anchor_box[3] - anchor_box[1] + 1)
  
  union_area = float(true_boxes_area + anchor_boxes_area - overlap_area)
  
  return overlap_area / union_area

def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [128, 128])

  return img

def create_boxes(boxes, horizontal):
  indices_list = {"small": [], "big": []}
  coords_list = {"small": [], "big": []}

  box_type = {"0.03125": "small", "0.0625": "big"}

  for box in boxes: # For each face in image
    box_x1 = box[0]
    box_y1 = box[1]

    box_x2 = box[2]
    box_y2 = box[3]

    if horizontal:
      w = box_x2 - box_x1
      box_x1 = (1 - box_x1) - w
      box_x2 = (box_x1 + w)

    face_box = np.array([box_x1, box_y1, box_x2, box_y2])

    iou = 0.01

    for box_values in list_of_boxes:
      best_indices = []
      best_coords = []

      for y_index, y_coord in enumerate(box_values): # y space
        for x_index, x_coord in enumerate(box_values): # x space we move throughout x, y keep its value and x change
          x1 = x_coord - box_values[0]
          y1 = y_coord - box_values[0]

          x2 = x_coord + box_values[0]
          y2 = y_coord + box_values[0]

          anchor_box = np.array([x1, y1, x2, y2])

          current_iou = compute_iou(face_box * 128.0, anchor_box * 128.0)

          if current_iou >= iou:
            iou = current_iou

            best_indices.append([x_index, y_index])
            best_coords.append(face_box)

      indices_list[box_type[str(box_values[0])]].extend(best_indices)
      coords_list[box_type[str(box_values[0])]].extend(best_coords)

  big_anchor = np.zeros((8, 8, 5))

  indices = indices_list["big"]
  coords = coords_list["big"]

  for index in range(len(indices)):
    x_index, y_index = indices[index]
    box = coords[index].tolist()
    big_anchor[y_index, x_index] = [1, *box]

  small_anchor = np.zeros((16, 16, 5))

  indices = indices_list["small"]
  coords = coords_list["small"]

  for index in range(len(indices)):
    x_index, y_index = indices[index]
    box = coords[index].tolist()
    small_anchor[y_index, x_index] = [1, *box]

  return big_anchor, small_anchor

def get_boxes_from_dictionary(image_path, horizontal):
  key = image_path.numpy().decode("utf-8") 
  boxes = boxes_dict[key]

  horizontal = bool(horizontal)

  big_anchor, small_anchor = create_boxes(boxes, horizontal)

  return big_anchor, small_anchor

def load_train_image(image_path):
  img = load_image(image_path)

  if tf.random.uniform([]) > 0.5:
    img = tf.image.random_saturation(img, 0.5, 1.5)

  if tf.random.uniform([]) > 0.5:
    img = tf.image.random_brightness(img, 0.2)

  horizontal = tf.random.uniform([]) > 0.5

  if horizontal:
    img = tf.image.flip_left_right(img)

  big_anchor, small_anchor = tf.py_function(get_boxes_from_dictionary, [image_path, horizontal], [tf.float32, tf.float32])

  return img, big_anchor, small_anchor, image_path, horizontal

def load_test_image(image_path):
  img = load_image(image_path)
  big_anchor, small_anchor = tf.py_function(get_boxes_from_dictionary, [image_path, False], [tf.float32, tf.float32])

  return image_path, img, big_anchor, small_anchor

test_dataset = tf.data.Dataset.from_tensor_slices((images_paths))
test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_generator = test_dataset.batch(32)

train_dataset = tf.data.Dataset.from_tensor_slices((images_paths))
train_dataset = train_dataset.shuffle(len(images_paths))
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_generator = train_dataset.batch(32)