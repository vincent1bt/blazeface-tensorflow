import numpy as np
import tensorflow as tf

small_boxes = np.linspace(0.03125, 0.96875, 16, endpoint=True, dtype=np.float32) # 16x16
big_boxes = np.linspace(0.0625, .9375, 8, endpoint=True, dtype=np.float32) # 8x8
list_of_boxes = [small_boxes, big_boxes]

small_x = tf.tile(tf.repeat(small_boxes, repeats=2), [16]) # x
small_y = tf.repeat(small_boxes, repeats=32)

small = tf.stack([small_x, small_y], axis=1)

big_x = tf.tile(tf.repeat(big_boxes, repeats=6), [8]) # x
big_y = tf.repeat(big_boxes, repeats=48)

big = tf.stack([big_x, big_y], axis=1)

reference_anchors = tf.concat([small, big], axis=0)

def mean_iou(true_boxes, pred_boxes, return_mean=True):
  x_min = tf.math.maximum(true_boxes[..., 0], pred_boxes[..., 0])
  y_min = tf.math.maximum(true_boxes[..., 1], pred_boxes[..., 1])
  x_max = tf.math.minimum(true_boxes[..., 2], pred_boxes[..., 2])
  y_max = tf.math.minimum(true_boxes[..., 3], pred_boxes[..., 3])
  
  overlap_area = tf.math.maximum(0.0, x_max - x_min + 1) * tf.math.maximum(0.0, y_max - y_min + 1)
  
  true_boxes_area = (true_boxes[..., 2] - true_boxes[..., 0] + 1) * (true_boxes[..., 3] - true_boxes[..., 1] + 1)
  
  predicted_boxes_area = (pred_boxes[..., 2] - pred_boxes[..., 0] + 1) * (pred_boxes[..., 3] - pred_boxes[..., 1] + 1)
  
  union_area = (true_boxes_area + predicted_boxes_area - overlap_area)

  if return_mean:
    return tf.math.reduce_mean(overlap_area / union_area)
  else:
    return overlap_area / union_area

def average_maximum_sup(class_predictions, anchor_predictions, reference_anchors, scale=128):
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

  for index, image_detections in enumerate(mask): # each 896 for each image
    # print(image_detections)
    num_image_detections = tf.keras.backend.sum(tf.dtypes.cast(image_detections, tf.int32))
    # print(image_detections.shape)
    # print(num_image_detections)

    if num_image_detections == 0:
      final_detections.append([])
    else:
      filtered_boxes = tf.boolean_mask(offset_boxes[index], image_detections)
      filtered_scores = tf.boolean_mask(class_predictions[index], image_detections)

      final_detections.append(tf.concat([tf.expand_dims(filtered_scores, axis=-1), filtered_boxes], axis=-1)) # num_image_detections, 5

  output_detections = []

  for image_detections in final_detections: # for each image B
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

      # mask = ious > 0.3 # 4

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


def order_big_anchors_randomly(big_anchors):
  big_anchor_list = []
  total_boxes = 30

  for batch_element in big_anchors: # B, 8, 8, 5
    box_index = tf.random.uniform([], 0, 6, dtype=tf.dtypes.int32) * 5 # 0, 5, 10, 15, 20, 25

    if box_index == 0:
      right_box = tf.zeros((8, 8, total_boxes - 5))

      anchor_tensor = tf.concat([batch_element, right_box], axis=-1)
      big_anchor_list.append(anchor_tensor)

    elif box_index == 25:
      left_box = tf.zeros((8, 8, total_boxes - 5))

      anchor_tensor = tf.concat([left_box, batch_element], axis=-1)
      big_anchor_list.append(anchor_tensor)

    else:
      left_box = tf.zeros((8, 8, box_index))
      right_box = tf.zeros((8, 8, total_boxes - box_index - 5))

      anchor_tensor = tf.concat([left_box, batch_element, right_box], axis=-1)
      big_anchor_list.append(anchor_tensor)

    
  big_anchor_list = tf.stack(big_anchor_list, axis=0) # B, 8, 8, 30

  return big_anchor_list


def order_small_anchors_randomly(small_anchors):
  small_anchor_list = []

  for batch_element in small_anchors: # B, 16, 16, 5
    box_index = tf.random.uniform([], 0, 2, dtype=tf.dtypes.int32) * 5 # 0, 5

    if box_index == 0:
      right_box = tf.zeros((16, 16, 5))

      anchor_tensor = tf.concat([batch_element, right_box], axis=-1)
      small_anchor_list.append(anchor_tensor)

    else:
      left_box = tf.zeros((16, 16, 5))

      anchor_tensor = tf.concat([left_box, batch_element], axis=-1)
      small_anchor_list.append(anchor_tensor)
    
  small_anchor_list = tf.stack(small_anchor_list, axis=0) # B, 16, 16, 10

  return small_anchor_list

