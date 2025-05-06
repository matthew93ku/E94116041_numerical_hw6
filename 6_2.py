import numpy as np

# 定義矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
])

# 計算逆矩陣
A_inv = np.linalg.inv(A)

# 輸出逆矩陣
print("A 的逆矩陣：")
print(A_inv)