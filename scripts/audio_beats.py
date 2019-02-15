"""playground music_beats
"""

import numpy as np
import matplotlib.pylab as plt

# v1: 2 patterns
# v2: 16 probs for event

t = np.linspace(0, 16-1, 16)

print("t", t)

p1 = np.zeros((16, 1))
p2 = np.zeros((16, 1))

p1[[0, 4, 8, 12],0] = 1.
p2[[0, 6, 8, 14],0] = 1.


plt.subplot(211)
plt.bar(t, p1) # , "ko")
plt.subplot(212)
plt.bar(t, p2) #, "ko")
# plt.gcf().adjust_subplots()
plt.show()
