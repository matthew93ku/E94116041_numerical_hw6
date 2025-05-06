import numpy as np

# 定義增廣矩陣 (A|b)
A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
])
b = np.array([1.12, 3.44, 2.15, 4.16])

# 解線性方程組 Ax = b
x = np.linalg.solve(A, b)

# 輸出結果
print("解為：")
for i, value in enumerate(x):
    print(f"x[{i+1}] = {value:.4f}")

# 驗證結果 (計算 Ax - b 是否接近零)
result = A @ x - b
print("\n驗證 (Ax - b)：")
print(result)