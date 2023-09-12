from Helipad import Helipad
import cv2
import numpy as np

tracker = Helipad()

image_path = './data/frame-0080.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

tracker.set_image(img)
tracker.set_color_mask(np.array([41, 109, 198]))

helipad_position_in_image = tracker.find_helipad()
img = tracker.draw_circle(helipad_position_in_image)

Helipad._show_image(img)