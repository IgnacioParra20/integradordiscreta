def numpy_benchmark(epochs=8000, lr=0.5):
    try:
        import numpy as np
    except ImportError:
        print("NumPy no está instalado. omitiendo comparación.")
        return
    W1 = np.array([[4.,4.],[-4.,-4.]], dtype=float)
    b1 = np.array([-2.,6.], dtype=float)
    W2 = np.array([[6.,6.]], dtype=float)
    b2 = np.array([-9.], dtype=float)

    X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=float)
    Y = np.array([[0.],[1.],[1.],[0.]], dtype=float)

    def s(x): return 1/(1+np.exp(-x))

    import time
    t0 = time.time()
    for _ in range(epochs):
        Z1 = X @ W1.T + b1
        A1 = s(Z1)
        Z2 = A1 @ W2.T + b2
        YH = s(Z2)
        dZ2 = YH - Y
        dW2 = dZ2.T @ A1
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2
        dZ1 = dA1 * (A1 * (1 - A1))
        dW1 = dZ1.T @ X
        db1 = dZ1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    t1 = time.time()
    print(f"[NumPy] Tiempo entrenamiento: {t1-t0:.3f}s con {epochs} épocas")
