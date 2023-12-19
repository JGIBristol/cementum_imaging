import pickle
import matplotlib.pyplot as plt

import multiProcStraight

# Read in a mask + image
with open("image_and_mask.pickle", "rb") as data:
    image, mask = pickle.load(data)

# Display them
fig, axis = plt.subplots()
axis.imshow(image, cmap="gray")
axis.imshow(mask, alpha=0.3)
plt.show()

# Straighten them with Elis' code
straightened = multiProcStraight.straightenTest(mask, image)

# Error message:
# Traceback (most recent call last):
#   File "/home/mh19137/cementum/test.py", line 17, in <module>
#     straightened = multiProcStraight.straightenTest(mask, image)
#   File "/home/mh19137/cementum/multiProcStraight.py", line 253, in straightenTest
#     min_a1_x, max_a1_x = min(a1[:,0]), max(a1[:,0])
# IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
