import os
import time
import argparse

import tensorflow as tf

parser = argparse.ArgumentParser()

from data_generators import images_paths, train_generator
from loss_functions import compute_loss

from utils import order_small_anchors_randomly, order_big_anchors_randomly, mean_iou, reference_anchors

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--continue_training', default=False, action='store_true', help='Use training checkpoints to continue training the model')

opts = parser.parse_args()

from models import BlazeModel

batch_size = 32
epochs = opts.epochs
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

train_steps = int(len(images_paths) / batch_size) + 1

model = None
model = BlazeModel()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

if opts.continue_training:
  print("loading training checkpoints: ")                   
  print(tf.train.latest_checkpoint(checkpoint_dir))
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

positive_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
background_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

background_accuracy_results = []
positive_accuracy_results = []
loss_results = []
iou_results = []

@tf.function
def train_step(imgs, big_anchors, small_anchors, reference_anchors, model):
  with tf.GradientTape() as tape:
    anchor_predictions, class_predictions = model(imgs, training=True)

    loss, filtered_pred_coords, filtered_true_coords, positive_class_predictions, background_class_predictions = compute_loss(class_predictions, anchor_predictions, big_anchors, small_anchors, reference_anchors)

  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss, filtered_pred_coords, filtered_true_coords, positive_class_predictions, background_class_predictions

def train(epochs):
  for epoch in range(epochs):
    batch_time = time.time()
    epoch_time = time.time()
    step = 0
    epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

    positive_accuracy_metric.reset_states()
    background_accuracy_metric.reset_states()

    for imgs, big_anchors, small_anchors, _, _ in train_generator:
      small_anchors = order_small_anchors_randomly(small_anchors)
      big_anchors = order_big_anchors_randomly(big_anchors)

      loss, filtered_pred_coords, filtered_true_coords, positive_class_predictions, background_class_predictions = train_step(imgs, big_anchors, small_anchors, reference_anchors, model)
      
      positive_accuracy_metric.update_state(tf.ones_like(positive_class_predictions), positive_class_predictions)
      background_accuracy_metric.update_state(tf.zeros_like(background_class_predictions), background_class_predictions)

      loss = float(loss)
      step += 1

      positive_accuracy = positive_accuracy_metric.result().numpy()
      background_accuracy = background_accuracy_metric.result().numpy()

      iou = mean_iou(filtered_true_coords * 128.0, filtered_pred_coords * 128.0)

      iou_results.append(iou)
      background_accuracy_results.append(background_accuracy)
      positive_accuracy_results.append(positive_accuracy)
      loss_results.append(loss)

      print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
            '| Loss:', f"{loss:.5f}", '| Positive Accuracy:', f"{positive_accuracy:.4f}",
            '| Background Accuracy:', f"{background_accuracy:.4f}",
            '| iou:', f"{iou:.4f}", "| Step Time:", f"{time.time() - batch_time:.2f}", end='')    
        
      batch_time = time.time()

    checkpoint.save(file_prefix=checkpoint_prefix)

    print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
          '| Loss:', f"{loss:.5f}", '| Positive Accuracy:', f"{positive_accuracy:.4f}",
          '| Background Accuracy:', f"{background_accuracy:.4f}",
          '| iou:', f"{iou:.4f}", "| Epoch Time:", f"{time.time() - epoch_time:.2f}")

train(epochs)

model.save("weights/saved_model/face", include_optimizer=False)