import numpy as np
import pandas as pd
import cv2

def get_vertical_new_axes(params, final_image_size=128):
    boxes_width, boxes_height, height, width, vertical_smaller, horizontal_smaller, space_up, space_down, space_left, space_right = params
    # First find horizontal accomodation
    if horizontal_smaller == "left":
        min_width = 0
        while True:
            # space we take from the left size
            box_left_space = np.random.randint(min_width, space_left + 1)
            current_box_size = box_left_space + boxes_width

            if current_box_size >= final_image_size:
                min_width_needed = 0
            else:
                min_width_needed = (final_image_size - current_box_size)

            if boxes_height >= current_box_size and boxes_height >= final_image_size:
                min_width_needed = (boxes_height - current_box_size)

            if space_right >= min_width_needed:
                break
            else:
                min_width = box_left_space

        # space we take from the right size
        box_right_space = np.random.randint(min_width_needed, space_right + 1)
        image_horizontal_space = box_left_space + boxes_width + box_right_space

    else:
        min_width = 0
        while True:
            box_right_space = np.random.randint(min_width, space_right + 1) # check what happen when i have the same number in both parameters
            current_box_size = box_right_space + boxes_width
            min_width_needed = 0 if current_box_size >= final_image_size else (final_image_size - current_box_size)

            if current_box_size >= final_image_size:
                min_width_needed = 0
            else:
                min_width_needed = (final_image_size - current_box_size)

            if boxes_height >= current_box_size and boxes_height >= final_image_size:
                min_width_needed = (boxes_height - current_box_size)

            if space_left >= min_width_needed:
                break
            else:
                min_width = box_right_space

        # space we take from the left size
        box_left_space = np.random.randint(min_width_needed, space_left + 1)
        image_horizontal_space = box_right_space + boxes_width + box_left_space

    # Then find vertical accomodation
    if vertical_smaller == "up":
        min_height = 0  
        max_height = space_up + 1

        while True:
            box_up_space = np.random.randint(min_height, max_height)
            current_box_size = box_up_space + boxes_height
            min_height_needed = image_horizontal_space - current_box_size # -10

            if current_box_size > image_horizontal_space:
                max_height = box_up_space
                continue

            if space_down >= min_height_needed:
                break
            else:
                # we need more space in the up size
                min_height = box_up_space

            # space we take from the down side
        box_down_space = min_height_needed #np.random.randint(min_height_needed, image_horizontal_space)
        image_vertical_space = box_up_space + boxes_height + box_down_space
        
    else:
        min_height = 0  
        max_height = space_down + 1

        while True:
            box_down_space = np.random.randint(min_height, max_height)
            current_box_size = box_down_space + boxes_height
            min_height_needed = image_horizontal_space - current_box_size

            if current_box_size > image_horizontal_space:
                # our current box height is higher than the wider box
                max_height = box_down_space
                continue

            if space_up >= min_height_needed:
                break
            else:
                # we need more space in the up size
                min_height = box_down_space

        # space we take from the left size
        box_up_space = min_height_needed #np.random.randint(min_height_needed, image_horizontal_space)
        image_vertical_space = box_down_space + boxes_height + box_up_space

    return image_vertical_space, image_horizontal_space, box_up_space, box_down_space, box_right_space, box_left_space

def get_horizontal_new_axes(params, final_image_size=128):
    boxes_width, boxes_height, height, width, vertical_smaller, horizontal_smaller, space_up, space_down, space_left, space_right = params
    borders = 0
    if boxes_width > height:
        diff = width - height

        if (diff // 2) < 100:
            borders = (diff // 2) - 2
        else:
            borders = 100
        
        space_up += borders
        space_down += borders

    # First find vertical acomodation
    if vertical_smaller == "up":
        min_height = 0

        while True:
            box_up_space = np.random.randint(min_height, space_up + 1)
            current_box_size = box_up_space + boxes_height
            
            # If the current space used is greater than the final image size
            if current_box_size >= final_image_size: # final_image_size = 128
                min_height_needed = 0
            else:
                min_height_needed = (final_image_size - current_box_size)

            # If our current space usage is less than the space used in the width
            # we need more space
            if boxes_width >= current_box_size and boxes_width >= final_image_size:
                min_height_needed = (boxes_width - current_box_size)

            if space_down >= min_height_needed:
                break
            else:
                min_height = box_up_space # use more space

        # space we take from the down side
        box_down_space = np.random.randint(min_height_needed, space_down + 1)
        image_vertical_space = box_up_space + boxes_height + box_down_space
        
    else:
        min_height = 0

        while True:
            box_down_space = np.random.randint(min_height, space_down + 1)
            current_box_size = box_down_space + boxes_height

            if current_box_size >= final_image_size: # final_image_size = 128
                min_height_needed = 0
            else:
                min_height_needed = (final_image_size - current_box_size)

            if boxes_width >= current_box_size and boxes_width >= final_image_size:
                min_height_needed = (boxes_width - current_box_size)

            if space_up >= min_height_needed:
                break
            else:
                min_height = box_down_space # use more space

        # space we take from the up side
        box_up_space = np.random.randint(min_height_needed, space_up + 1)
        image_vertical_space = box_down_space + boxes_height + box_up_space
        
    # Then horizontal accomodation
    if horizontal_smaller == "left":
        min_width = 0
        max_width = space_left + 1

        while True:
            # space we take from the left size
            box_left_space = np.random.randint(min_width, max_width)
            current_box_size = box_left_space + boxes_width
            min_width_needed = image_vertical_space - current_box_size # -10

            if current_box_size > image_vertical_space:
                max_width = box_left_space
                continue

            if space_right >= min_width_needed:
                break
            else:
                # we need more space in the up size
                min_width = box_left_space

        # space we take from the right size
        box_right_space = min_width_needed
        image_horizontal_space = box_left_space + boxes_width + box_right_space

    else:
        min_width = 0
        max_width = space_right + 1

        while True:
            box_right_space = np.random.randint(min_width, max_width)
            current_box_size = box_right_space + boxes_width
            min_width_needed = image_vertical_space - current_box_size

            if current_box_size > image_vertical_space:
                max_width = box_right_space
                continue

            if space_left >= min_width_needed:
                break
            else:
                min_width = box_right_space

        # space we take from the left size
        box_left_space = min_width_needed
        image_horizontal_space = box_right_space + boxes_width + box_left_space

    return image_vertical_space, image_horizontal_space, box_up_space, box_down_space, box_right_space, box_left_space, borders


def get_box_sizes(boxes):
  max_box_height = 0
  min_box_height = 99999

  max_box_width = 0
  min_box_width = 99999

  for box in boxes: # For each face in image
    x1 = box[0]
    y1 = box[1]

    x2 = (box[0] + box[2])
    y2 = (box[1] + box[3])

    current_min_box_height = y1
    current_max_box_height = y2

    if min_box_height > current_min_box_height:
        min_box_height = current_min_box_height

    if current_max_box_height > max_box_height:
        max_box_height = current_max_box_height

    current_min_box_width = x1
    current_max_box_width = x2

    if min_box_width > current_min_box_width:
        min_box_width = current_min_box_width

    if current_max_box_width > max_box_width:
        max_box_width = current_max_box_width

  return max_box_height, min_box_height, max_box_width, min_box_width

def get_random_image(img, boxes):
  H, W, C = img.shape

  # Get the corners of all the boxes to know where we can cut the image
  max_box_height, min_box_height, max_box_width, min_box_width = get_box_sizes(boxes)

  boxes_height = max_box_height - min_box_height
  boxes_width = max_box_width - min_box_width

  space_up = min_box_height
  space_down = H - max_box_height

  space_left = min_box_width
  space_right = W - max_box_width

  horizontal_smaller = "left" if space_right > space_left else "right"
  vertical_smaller = "up" if space_down > space_up else "down"

  small_direction = "horizontal" if H >= W else "vertical"

  params = [boxes_width, boxes_height, H, W, vertical_smaller, horizontal_smaller, space_up, space_down, space_left, space_right]

  if small_direction == "vertical":
      image_vertical_space, image_horizontal_space, box_up_space, box_down_space, box_right_space, box_left_space, borders = get_horizontal_new_axes(params)
  else:
      image_vertical_space, image_horizontal_space, box_up_space, box_down_space, box_right_space, box_left_space = get_vertical_new_axes(params)
      borders = 0

  image_x1 = min_box_width - box_left_space 
  image_x2 = max_box_width + box_right_space

  image_y1 = (min_box_height + borders) - box_up_space
  image_y2 = (max_box_height + borders) + box_down_space

  if borders > 0:
    new_img = np.zeros([H + (borders * 2), W, C], dtype=np.uint8)
    H, W, C = new_img.shape

    new_img[borders:H - borders, :, :] = img
    img = new_img

  resized = img[image_y1:image_y2, image_x1:image_x2, :]

  H, W, C = resized.shape

  try:
    resized = cv2.resize(resized, (128, 128), interpolation=cv2.INTER_AREA)
  except:
    print(resized.shape)
    print(image_x1, image_y1, image_x2, image_y2)
    print(borders)
    print(min_box_height)
    print(max_box_height, "max_box_height")
    print(box_down_space, "box_down_space")

    raise Exception("Image error")

  new_boxes = []

  for box in boxes: # For each face in image
    x1 = box[0] 
    y1 = box[1]

    x1 = (x1 - image_x1)
    y1 = ((y1 + borders) - image_y1)

    x2 = (x1 + box[2]) / W
    y2 = (y1 + box[3]) / H

    x1 = x1 / W
    y1 = y1 / H

    new_boxes.append([x1, y1, x2, y2])
  
  return resized, new_boxes


images_paths = []
boxes_dict = {}

df = pd.read_csv("data_files/fixed_images.csv", index_col=0)
face_df = df[['group', 'image_path', 'x1', 'y1', 'w', 'h']]

for img_path, indices in face_df.groupby("image_path").groups.items():
  selected = face_df.loc[indices]
  group = selected.values[0][0]
    
  original_img_path = f"face_dataset/{img_path}"
  original_img = cv2.imread(original_img_path)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if group == "None":
    img_path = img_path.replace("/", "-")
  else:
    img_path = img_path.split("/")[-1]

  img_path = f"created_dataset/set-1-{img_path}"

  boxes = selected.values[:, 2:]

  images_paths.append(img_path)

  resized, new_boxes = get_random_image(original_img, boxes)

  boxes_dict[img_path] = []

  for box in new_boxes:
    x1 = box[0] 
    y1 = box[1]

    x2 = box[2]
    y2 = box[3]
    
    boxes_dict[img_path].append(np.array([x1, y1, x2, y2], dtype=np.float32))
  
  cv2.imwrite(img_path, resized)

  if len(images_paths) % 500 == 0:
    print(len(images_paths))

for img_path, indices in face_df.groupby("image_path").groups.items():
  selected = face_df.loc[indices]
  group = selected.values[0][0]
    
  original_img_path = f"face_dataset/{img_path}"
  original_img = cv2.imread(original_img_path)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if group == "None":
    img_path = img_path.replace("/", "-")
  else:
    img_path = img_path.split("/")[-1]

  img_path = f"created_dataset/set-2-{img_path}"

  boxes = selected.values[:, 2:]

  images_paths.append(img_path)

  resized, new_boxes = get_random_image(original_img, boxes)

  boxes_dict[img_path] = []

  for box in new_boxes:
    x1 = box[0] 
    y1 = box[1]

    x2 = box[2]
    y2 = box[3]
    
    boxes_dict[img_path].append(np.array([x1, y1, x2, y2], dtype=np.float32))
  
  cv2.imwrite(img_path, resized)

  if len(images_paths) % 500 == 0:
    print(len(images_paths))

print(len(images_paths))

np.save("data_files/images_paths.npy", np.array(images_paths))
np.save("data_files/boxes_dict.npy", boxes_dict)