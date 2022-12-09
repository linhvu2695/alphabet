import matplotlib.pyplot as plt
import numpy as np

labels = np.load("data/labels.npy")

data = {}
for label in labels:
    data[label] = data.get(label, 0) + 1

data = sorted(data.items(), key=lambda x: x[1], reverse=True)
x = [d[0] for d in data]
y = [d[1] for d in data]

print("Number of classes: %d" %len(data))
print("Number of data points: %d" %len(labels))

plt.bar(x, y)
plt.show()