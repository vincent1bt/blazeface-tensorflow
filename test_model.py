import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_generators import test_generator
from utils import mean_iou, reference_anchors

def filter_boxes(class_predictions, anchor_predictions, reference_anchors, scale=128):
  # class_predictions # B, 896, 1
  # anchor_predictions # B, 896, 4

  x_center = reference_anchors[:, 0:1] + (anchor_predictions[..., 0:1] / scale) # 8, 896, 1
  y_center = reference_anchors[:, 1:2] + (anchor_predictions[..., 1:2] / scale) # 8, 896, 1

  w = anchor_predictions[..., 2:3] / scale # B, 896, 1
  h = anchor_predictions[..., 3:4] / scale # B, 896, 1

  y_min = y_center - h / 2.  # ymin
  x_min = x_center - w / 2.  # xmin
  y_max = y_center + h / 2.  # ymax
  x_max = x_center + w / 2.  # xmax

  offset_boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1) # B, 896, 4

  class_predictions = tf.squeeze(class_predictions, axis=-1)

  mask = class_predictions >= 0.75 # 0.75 B, 896

  final_detections = [] # final shape B, num_image_detections, 5 where num_image_detections can vary by image

  for index, image_detections in enumerate(mask): # each 896 for every image
    num_image_detections = tf.keras.backend.sum(tf.dtypes.cast(image_detections, tf.int32))

    if num_image_detections == 0:
      final_detections.append([])
    else:
      filtered_boxes = tf.boolean_mask(offset_boxes[index], image_detections)
      filtered_scores = tf.boolean_mask(class_predictions[index], image_detections)

      final_detections.append(tf.concat([tf.expand_dims(filtered_scores, axis=-1), filtered_boxes], axis=-1)) # num_image_detections, 5

  output_detections = []

  for image_detections in final_detections: # for each image in batch B
    # num_image_detections, 5
    if image_detections == []:
      output_detections.append([])
      continue

    remaining = tf.argsort(image_detections[:, 0], axis=0, direction='DESCENDING') # num_image_detections

    faces = []

    while remaining.shape[0] > 0:
      detection = image_detections[remaining[0]]
      first_box = detection[1:] # 1, 4
      other_boxes = tf.gather(image_detections, remaining)[:, 1:] # 4, 4

      ious = mean_iou(np.array(first_box) * 128.0, np.array(other_boxes) * 128.0, return_mean=False) # num_image_detections

      overlapping = tf.boolean_mask(remaining, ious > 0.3)
      remaining = tf.boolean_mask(remaining, ious <= 0.3) # When all false, returns shape 0
      # The remaining boxes should belong to a different face

      if overlapping.shape[0] > 1:
        overlapping_boxes = tf.gather(image_detections, overlapping)
        coordinates = overlapping_boxes[:, 1:] # overlapping, 4
        scores = overlapping_boxes[:, 0:1] # overlapping, 1
        total_score = tf.keras.backend.sum(scores)
        
        weighted_boxes = tf.keras.backend.sum((coordinates * scores), axis=0) / total_score # overlapping, 4
        weighted_score = total_score / overlapping.shape[0] # overlapping, 1

        weighted_score = tf.reshape(weighted_score, (1,))

        weighted_detection = tf.concat([weighted_score, weighted_boxes], axis=0) # overlapping, 5

        faces.append(weighted_detection)

      else:
        faces.append(detection)
      
    output_detections.append(faces)

  return output_detections, final_detections

def filter_true_boxes(class_predictions, anchor_predictions, reference_anchors, scale=128):
  # class_predictions # B, 320, 1
  # anchor_predictions # B, 320, 4

  offset_boxes = anchor_predictions
  class_predictions = tf.squeeze(class_predictions, axis=-1)
  mask = class_predictions >= 0.75 # B, 320

  final_detections = [] # final shape B, num_image_detections, 5 where num_image_detections can vary by image

  for index, image_detections in enumerate(mask): # each 320 for every image
    num_image_detections = tf.keras.backend.sum(tf.dtypes.cast(image_detections, tf.int32))

    if num_image_detections == 0:
      final_detections.append([])
    else:
      filtered_boxes = tf.boolean_mask(offset_boxes[index], image_detections)
      filtered_scores = tf.boolean_mask(class_predictions[index], image_detections)

      final_detections.append(tf.concat([tf.expand_dims(filtered_scores, axis=-1), filtered_boxes], axis=-1)) # num_image_detections, 5

  output_detections = []

  for image_detections in final_detections: # for each image in batch B
    # num_image_detections, 5
    if image_detections == []:
      output_detections.append([])
      continue

    remaining = tf.argsort(image_detections[:, 0], axis=0, direction='DESCENDING') # num_image_detections

    faces = []

    while remaining.shape[0] > 0:
      detection = image_detections[remaining[0]]
      first_box = detection[1:] # 1, 4
      other_boxes = tf.gather(image_detections, remaining)[:, 1:] # 4, 4

      ious = mean_iou(np.array(first_box) * 128.0, np.array(other_boxes) * 128.0, return_mean=False) # num_image_detections

      overlapping = tf.boolean_mask(remaining, ious > 0.8)
      remaining = tf.boolean_mask(remaining, ious <= 0.8) # When all false, returns shape 0
      # The remaining boxes should belong to a different face

      if overlapping.shape[0] > 1:
        overlapping_boxes = tf.gather(image_detections, overlapping)
        coordinates = overlapping_boxes[:, 1:] # overlapping, 4
        scores = overlapping_boxes[:, 0:1] # overlapping, 1
        total_score = tf.keras.backend.sum(scores)
        
        weighted_boxes = tf.keras.backend.sum((coordinates * scores), axis=0) / total_score # overlapping, 4
        weighted_score = total_score / overlapping.shape[0] # overlapping, 1

        weighted_score = tf.reshape(weighted_score, (1,))

        weighted_detection = tf.concat([weighted_score, weighted_boxes], axis=0) # overlapping, 5

        faces.append(weighted_detection)

      else:
        faces.append(detection)
      
    output_detections.append(faces)

  return output_detections, final_detections

def get_final_boxes(imgs, big_anchors, small_anchors, reference_anchors, model):
  B = big_anchors.shape[0]
  list_big_anchors = tf.reshape(big_anchors, (B, -1, 5))
  list_small_anchors = tf.reshape(small_anchors, (B, -1, 5))

  list_true_anchors = tf.concat([list_small_anchors, list_big_anchors], axis=1) # shape (B, 320, 5)

  true_classes = list_true_anchors[:, :, :1] # shape (B, 320, 1)
  true_coords = list_true_anchors[:, :, 1:] # shape (B, 320, 1)

  anchor_predictions, class_predictions = model.predict(imgs)

  output_detections, _ = filter_true_boxes(true_classes, true_coords, reference_anchors)
  predicted_output_detections, _ = filter_boxes(class_predictions, anchor_predictions, reference_anchors)

  return output_detections, predicted_output_detections

def test_model(output_detections, predicted_output_detections, imgs, size=8):
  fig, ax = plt.subplots(2, 4, figsize=(15, 9))

  for index in range(size):
    y = 0 if index < 4 else 1
    x = index if index < 4 else index - 4

    true_boxes = output_detections[index]
    pred_boxes = predicted_output_detections[index]
    img = imgs[index].numpy()

    for box in true_boxes: 
      x1 = int(box[1] * 128)
      y1 = int(box[2] * 128)
      x2 = int(box[3] * 128)
      y2 = int(box[4] * 128)
      color = (0, 0, 255)

      img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    score = 1

    if pred_boxes != []:
      for box in pred_boxes:
        score = box[0]
        x1 = int(box[1] * 128)
        y1 = int(box[2] * 128)
        x2 = int(box[3] * 128)
        y2 = int(box[4] * 128)
        color = (255, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    ax[y, x].imshow(img)
    ax[y, x].set_title('Score {}'.format(score))

model = tf.keras.models.load_model('weights/saved_model/face')

for image_paths, imgs, big_anchors, small_anchors in test_generator:
  break

output_detections, predicted_output_detections = get_final_boxes(imgs, big_anchors, small_anchors, reference_anchors, model)
test_model(output_detections[0:8], predicted_output_detections[0:8], imgs[0:8])