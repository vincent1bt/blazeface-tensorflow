import tensorflow as tf

huber_loss = tf.keras.losses.Huber()

def compute_loss(class_predictions, anchor_predictions, big_anchors, small_anchors, reference_anchors, ratio=3, scale=128):
  B = big_anchors.shape[0]
  list_big_anchors = tf.reshape(big_anchors, (B, -1, 5)) # shape [B, 384, 5])

  list_small_anchors = tf.reshape(small_anchors, (B, -1, 5)) # shape [B, 512, 5])

  list_true_anchors = tf.concat([list_small_anchors, list_big_anchors], axis=1) # shape [B, 896, 5]

  true_classes = list_true_anchors[:, :, 0] # shape [B, 896, 1]
  true_coords = list_true_anchors[:, :, 1:] # shape [B, 896, 4]

  faces_mask_bool = tf.dtypes.cast(true_classes, tf.bool)

  faces_num = tf.keras.backend.sum(true_classes)
  background_num = int(faces_num * ratio) // B

  class_predictions = tf.squeeze(class_predictions, axis=-1)

  # Hard negatives

  predicted_classes_scores = tf.where(faces_mask_bool, -99.0, class_predictions) # B, 896

  background_class_predictions = tf.sort(predicted_classes_scores, axis=-1, direction='DESCENDING')[:, :background_num]
  positive_class_predictions = tf.boolean_mask(class_predictions, faces_mask_bool)

  # Class loss

  background_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(background_class_predictions), background_class_predictions))
  positive_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(positive_class_predictions), positive_class_predictions))

  # Anchors offset

  # anchor_predictions (shape) B, 894, 4

  x_center = reference_anchors[:, 0:1] + (anchor_predictions[..., 0:1] / scale) # 8, 896, 1
  y_center = reference_anchors[:, 1:2] + (anchor_predictions[..., 1:2] / scale) # 8, 896, 1

  w = anchor_predictions[..., 2:3] / scale # B, 896, 1
  h = anchor_predictions[..., 3:4] / scale # B, 896, 1

  y_min = y_center - h / 2.  # ymin
  x_min = x_center - w / 2.  # xmin
  y_max = y_center + h / 2.  # ymax
  x_max = x_center + w / 2.  # xmax

  offset_boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1) # B, 896, 4

  filtered_pred_coords = tf.boolean_mask(offset_boxes, faces_mask_bool) # ~faces_num, 4
  filtered_true_coords = tf.boolean_mask(true_coords, faces_mask_bool) # ~faces_num, 4.

  detection_loss = huber_loss(filtered_true_coords, filtered_pred_coords)

  loss = tf.math.reduce_mean(detection_loss) * 150 + (background_loss * 35) + (positive_loss * 35)

  return loss, filtered_pred_coords, filtered_true_coords, positive_class_predictions, background_class_predictions