import numpy as np
from time import time
from scipy.sparse.linalg import svds
import pickle


class AkMatrixMaker:
    def __init__(self):
        self.A_matrix = None
        self.A_k_matrix = None
        self.number_of_texts = 0
        self.d_vectors_len = 0

    def save_A_k_matrix(self, A_k_matrix_file):
        time_ = time()
        with open(A_k_matrix_file, "wb") as f:
            pickle.dump(self.A_k_matrix, f)
        print(f"saving time: {time() - time_}")

    def create_A_k_matrix(self, k=10):
        time_ = time()
        k = min(k, min(self.A_matrix.shape) - 1)
        U, S, V = svds(self.A_matrix, k=k)
        A_k = np.zeros_like(self.A_matrix).astype('float64')
        for i in range(k):
            A_k += np.outer(U[:, i], V[i, :]) * S[i]

        for i in range(self.number_of_texts):
            if np.linalg.norm(A_k[:, i]):
                A_k[:, i] = A_k[:, i] / np.linalg.norm(A_k[:, i])

        self.d_vectors_len = len(self.d_vectors)
        self.d_vectors = None
        self.A_k_matrix = A_k
        print(f"creating time: {time() - time_}")


if __name__ == '__main__':
    SAVED_CLASS_PATH = "C:\\data.pkl"
    A_K_MATRIX_PATH = "C:\\A_k_matrix_3000.txt"
    with open(SAVED_CLASS_PATH, 'rb') as inp:
        AkMatrixMaker = pickle.load(inp)

    AkMatrixMaker.create_A_k_matrix(5000)
    AkMatrixMaker.save_A_k_matrix(A_K_MATRIX_PATH)
