import numpy as np
from time import time
from nltk.stem import PorterStemmer
import pickle


def print_search_results(search_results):
    for result in search_results:
        print(f"link: {result[2]}"
              f" similarity: {result[1]}")


def without_SVD(words, k=3):
    time_ = time()
    search_results = SE.search(words, k=k, similarity=2)
    print(f"without SVD, search time: {time() - time_}")
    print_search_results(search_results)


def with_SVD(words, k=3):
    time_ = time()
    search_results = SE.search(words, k=k, similarity=3)
    print(f"with SVD, search time: {time() - time_}")
    print_search_results(search_results)


class SearchEngine:
    def __init__(self):
        self.d_vectors = None
        self.d_vectors_len = 0
        self.A_k_matrix = None
        self.texts_with_indexes = None
        self.A_matrix = None
        self.similarity = None
        self.words_size = None
        self.words = None
        self.IDF_vector = None

    def load_A_k_matrix(self):
        with open(SAVED_A_K_MATRIX_PATH, "rb") as f:
            self.A_k_matrix = pickle.load(f)

    def similarity2(self, q_vector_org):
        q_vector = q_vector_org.copy()

        if np.linalg.norm(q_vector) != 0:
            q_vector = q_vector / np.linalg.norm(q_vector)
        similarity = (q_vector @ self.A_matrix)[0]

        correlation_matrix = [(self.texts_with_indexes[i][1],
                               similarity[i],
                               self.texts_with_indexes[i][0])
                              for i in range(len(self.d_vectors))]

        return correlation_matrix

    def similarity3(self, q_vector_org):
        q_vector = q_vector_org.copy()
        if np.linalg.norm(q_vector) != 0:
            q_vector = q_vector / np.linalg.norm(q_vector)

        similarity = (q_vector @ self.A_k_matrix)[0]

        correlation_matrix = [(self.texts_with_indexes[i][1],
                               similarity[i],
                               self.texts_with_indexes[i][0])
                              for i in range(len(self.d_vectors))]

        return correlation_matrix

    def search(self, words, k=3, similarity=3):
        if similarity == 2:
            self.similarity = self.similarity2
        else:
            self.similarity = self.similarity3

        k = min(k, 100)
        q_vector = np.zeros((1, self.words_size))

        for word in words:
            stemmed_word = ps.stem(word)
            if stemmed_word in self.words:
                idx = self.words[stemmed_word][0]
                q_vector[0][idx] += 1

        q_vector *= self.IDF_vector

        correlation_matrix = self.similarity(q_vector)
        correlation_matrix.sort(key=lambda x: -x[1])
        return correlation_matrix[:k]


if __name__ == '__main__':
    time_ = time()
    print("Loading Search Engine...")
    ps = PorterStemmer()
    SAVED_CLASS_PATH = "C:\\data.pkl"
    SAVED_A_K_MATRIX_PATH = "C:\\A_k_matrix.txt"

    with open(SAVED_CLASS_PATH, 'rb') as inp:
        SE = pickle.load(inp)
    SE.load_A_k_matrix(SAVED_A_K_MATRIX_PATH)
    print(f"loading time: {time() - time_}")
    print("Search Engine is ready to use!")

    while 1:
        try:
            words = input("\nWords to search: ").split()
            num_of_top_results = input("How many of top results to show: ")
            while not num_of_top_results.isnumeric():
                print("Wrong number!")
                num_of_top_results = input("How many of top results to show: ")
            num_of_top_results = int(num_of_top_results)

            mode = input("Mode (1 -> both), "
                         "(2 -> without SVD), "
                         "(3 -> with SVD): ")
            while not 1 <= int(mode) <= 3:
                print("Wrong mode!")
                mode = input("Mode (1 -> both), "
                             "(2 -> without SVD), "
                             "(3 -> with SVD): ")

            if int(mode) == 1:
                without_SVD(words, num_of_top_results)
                with_SVD(words, num_of_top_results)
            elif int(mode) == 2:
                without_SVD(words, num_of_top_results)
            else:
                with_SVD(words, num_of_top_results)
        except KeyboardInterrupt:
            print("ending program...")
            SE = None
            break
