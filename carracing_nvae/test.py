import cv2

import numpy as np

a = np.load("/dataset/imgs/49072220074.npz")["obs"]

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('output.avi', fourcc, 50.0, (84, 96))

for i in range(a.shape[0]):
    out.write(a[i])

out.release()
