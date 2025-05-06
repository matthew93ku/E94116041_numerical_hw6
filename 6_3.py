import numpy as np

# 定義矩陣 A 和向量 b
A = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)
b = np.array([2, 3, 4, 1], dtype=float)

# 初始化 L 和 U
n = A.shape[0]
L = np.zeros((n, n))
U = np.zeros((n, n))

# Crout 分解
# 設置 U 的對角線元素為 1
for i in range(n):
    U[i, i] = 1

# 計算 L 和 U
# 第一行
L[0, 0] = A[0, 0]  # l_{11} = a_{11}
U[0, 1] = A[0, 1] / L[0, 0]  # u_{12} = a_{12} / l_{11}

# 中間行
for i in range(1, n-1):
    L[i, i-1] = A[i, i-1]  # l_{i,i-1} = a_{i,i-1}
    L[i, i] = A[i, i] - L[i, i-1] * U[i-1, i]  # l_{ii} = a_{ii} - l_{i,i-1} * u_{i-1,i}
    U[i, i+1] = A[i, i+1] / L[i, i]  # u_{i,i+1} = a_{i,i+1} / l_{ii}

# 最後一行
L[n-1, n-2] = A[n-1, n-2]  # l_{n,n-1} = a_{n,n-1}
L[n-1, n-1] = A[n-1, n-1] - L[n-1, n-2] * U[n-2, n-1]  # l_{nn} = a_{nn} - l_{n,n-1} * u_{n-1,n}

# 解 L y = b（前向代入）
y = np.zeros(n)
for i in range(n):
    y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]

# 解 U x = y（後向代入）
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = y[i] - np.sum(U[i, i+1:] * x[i+1:])

# 輸出結果
print("解為：")
for i in range(n):
    print(f"x[{i+1}] = {x[i]:.4f}")

# 驗證
print("\n驗證 (Ax - b)：")
print(A @ x - b)