import pickle
import matplotlib.pyplot as plt

import multiProcStraight

# Read in a mask + image
with open("cementum_data.pickle", "rb") as data:
    _, _, test_images, test_masks = pickle.load(data)
image, mask = test_images[0], test_masks[0]

# Straighten them with Elis' code
straight = multiProcStraight.straightenTest(mask, image)
plt.imshow(straight)

plt.show()
