import numpy as np

arr = np.array([
    [
        [1]
    ],
    [
        [2]
    ],
    [
        [3]
    ]
])

print(arr)

arr[arr[:, :] > 2] = 0

print(arr)