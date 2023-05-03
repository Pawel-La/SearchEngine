import numpy as np
from time import time
from os.path import join
import spacy
from concurrent.futures import ProcessPoolExecutor
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse.linalg import svds


def get_file_idx(file_path):
    return int(file_path.split("\\")[-1][:-4])


def get_tokens_and_index(saved_file):
    idx = get_file_idx(saved_file)

    try:
        with open(saved_file, 'r', encoding='utf-8') as file:
            text_ = nlp(file.read()[500:20_500])
            tokens = []
            for token in text_:
                string = token.text
                string = string.lower()
                if string.isalpha():
                    tokens.append(ps.stem(string))
            return tokens, idx
    except FileNotFoundError | FileExistsError:
        return None, None


def get_all_tokens_and_indexes():
    results = []
    with ProcessPoolExecutor() as executor:
        for tokens, idx in executor.map(get_tokens_and_index, saved_files):
            if tokens and idx:
                results.append((tokens, idx))
    return results


class SearchEngine:
    def __init__(self, texts_with_indexes):
        self.texts_with_indexes = texts_with_indexes
        self.number_of_texts = len(texts_with_indexes)
        self.words = {}
        self.words_size = 0
        self.d_vectors = []
        self.A_matrix = None
        self.IDF_vector = None
        self.similarity = None
        self.preprocess()
        time_ = time()
        self.read_data()
        print("reading time: ", time() - time_)
        self.create_matrix()

    def preprocess(self):
        time_ = time()
        self.create_words_dict()
        self.create_IDF_vector()
        self.create_d_vectors()
        print("preprocessing time: ", time() - time_)
        time_ = time()
        self.save_data()
        print("saving time: ", time() - time_)

    def create_words_dict(self):
        idx = 0
        for text, _ in self.texts_with_indexes:
            for word in text:
                if word not in self.words:
                    self.words[word] = (idx, 0)
                    idx += 1

        for text, _ in self.texts_with_indexes:
            n_vector = np.zeros(len(self.words))
            for word in text:
                if word in self.words:
                    idx = self.words[word][0]
                    if not n_vector[idx]:
                        self.words[word] = (
                            self.words[word][0],
                            self.words[word][1] + 1,
                        )
                        n_vector[idx] = 1

        print(f"size before deletion: {len(self.words)}")

        keys_to_delete = []
        for key, value in self.words.items():
            if value[1] <= 10 or value[1] >= self.number_of_texts / 10:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            self.words.pop(key)

        for word in stopwords.words('english'):
            if word in self.words:
                self.words.pop(word)

        l = sorted(list(self.words.items()), key=lambda x: x[1][1])
        number_of_keys_to_delete = max(0, len(self.words) - 10_000)
        keys_to_delete = []
        for i in range(number_of_keys_to_delete):
            keys_to_delete.append(l[i][0])
        for key in keys_to_delete:
            self.words.pop(key)

        idx = 0
        for key, value in self.words.items():
            self.words[key] = (idx, value[1])
            idx += 1

        self.words_size = len(self.words)
        print(f"size after deletion: {len(self.words)}")

    def create_IDF_vector(self):
        l = list(self.words.values())
        l.sort()
        self.IDF_vector = np.zeros(self.words_size)

        for idx, value in l:
            self.IDF_vector[idx] = np.log(self.number_of_texts / value)

    def create_d_vectors(self):
        vectors_file = \
            "C:\\Users\\Paweł\\Documents\\mownit\\vectors\\vectors.txt"
        with open(vectors_file, 'w', encoding='utf-8') as file:
            file.write("")

        for text, _ in self.texts_with_indexes:
            d_vector = np.zeros(self.words_size)
            for word in text:
                if word in self.words:
                    idx = self.words[word][0]
                    d_vector[idx] += 1

            d_vector *= self.IDF_vector

            with open(vectors_file, 'a', encoding='utf-8') as file:
                np.savetxt(file, d_vector, newline=' ', encoding='utf-8')
                file.write("\n")

    def create_matrix(self):
        self.A_matrix = np.zeros((self.words_size, self.number_of_texts))
        for count, d_vector in enumerate(self.d_vectors):
            self.A_matrix[:, count] = d_vector

    def save_data(self):
        self.save_texts()
        self.save_words()

    def save_texts(self):
        texts_file = "C:\\Users\\Paweł\\Documents\\mownit\\texts\\texts.txt"
        with open(texts_file, 'w', encoding='utf-8') as file:
            file.write("")
        for _, link_idx in self.texts_with_indexes:
            with open(texts_file, 'a', encoding='utf-8') as file:
                file.write(f"{link_idx}\n")
        self.texts_with_indexes = None

    def save_words(self):
        words_file = "C:\\Users\\Paweł\\Documents\\mownit\\words\\words.txt"
        with open(words_file, 'w', encoding='utf-8') as file:
            file.write("")
        for key, value in self.words.items():
            with open(words_file, 'a', encoding='utf-8') as file:
                file.write(f"{key} {value[0]} {value[1]}\n")
        self.words = None

    def read_data(self):
        self.read_texts()
        self.read_words()
        self.read_vectors()

    def read_texts(self):
        texts_path = "C:\\Users\\Paweł\\Documents\\mownit\\texts\\texts.txt"
        with open(texts_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
        self.texts_with_indexes = [(0, int(line[:-1])) for line in lines]

    def read_words(self):
        words = {}
        words_path = "C:\\Users\\Paweł\\Documents\\mownit\\words\\words.txt"
        with open(words_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            split_line = line[:-1].split()
            words[split_line[0]] = (int(split_line[1]), int(split_line[2]))
        self.words = words

    def read_vectors(self):
        vectors_path = \
            "C:\\Users\\Paweł\\Documents\\mownit\\vectors\\vectors.txt"
        matrix = np.loadtxt(vectors_path, dtype=np.float64, encoding='utf-8')
        d_vectors = []
        for i in range(len(matrix)):
            d_vectors.append(matrix[i])
        self.d_vectors = d_vectors

    def similarity1(self, q_vector):
        correlation_matrix = []
        for i in range(len(self.d_vectors)):
            if np.linalg.norm(q_vector) == 0:
                correlation_matrix.append((i, 0))
            else:
                value = (q_vector @ (self.d_vectors[i]).T) / \
                        (
                        np.linalg.norm(q_vector) *
                        np.linalg.norm(self.d_vectors[i])
                        )
                correlation_matrix.append((self.texts_with_indexes[i][1],
                                           value[0]))
        return correlation_matrix

    def similarity2(self, q_vector_org):
        correlation_matrix = []
        d_vectors = self.d_vectors.copy()
        q_vector = q_vector_org.copy()

        for i in range(len(d_vectors)):
            d_vectors[i] = d_vectors[i] / np.linalg.norm(d_vectors[i])

        A_matrix = np.zeros((self.words_size, self.number_of_texts))
        for count, d_vector in enumerate(d_vectors):
            A_matrix[:, count] = d_vector

        if np.linalg.norm(q_vector) != 0:
            q_vector = q_vector / np.linalg.norm(q_vector)
        result = (q_vector @ A_matrix)[0]

        for i in range(len(self.d_vectors)):
            correlation_matrix.append((self.texts_with_indexes[i][1],
                                       result[i]))

        return correlation_matrix

    def similarity3(self, q_vector_org, k=1000):
        k = min(k, min(self.A_matrix.shape) - 1)
        correlation_matrix = []
        q_vector = q_vector_org.copy()

        U, S, V = svds(self.A_matrix, k=k)
        A_k = np.zeros_like(self.A_matrix).astype('float64')
        for i in range(k):
            A_k += np.outer(U[:, i], V[i, :]) * S[i]

        for i in range(self.number_of_texts):
            A_k[:, i] = A_k[:, i] / np.linalg.norm(A_k[:, i])

        if np.linalg.norm(q_vector) != 0:
            q_vector = q_vector / np.linalg.norm(q_vector)
        result = (q_vector @ A_k)[0]

        for i in range(len(self.d_vectors)):
            correlation_matrix.append((self.texts_with_indexes[i][1],
                                       result[i]))

        return correlation_matrix

    def search(self, words, k=2, similarity=3):
        if similarity == 1:
            self.similarity = self.similarity1
        elif similarity == 2:
            self.similarity = self.similarity2
        else:
            self.similarity = self.similarity3

        k = min(k, self.number_of_texts)
        q_vector = np.zeros((1, self.words_size))

        time_ = time()
        for word in words:
            stemmed_word = ps.stem(word)
            if stemmed_word in self.words:
                idx = self.words[stemmed_word][0]
                q_vector[0][idx] += 1
        print("created and filled q vector time: ", time() - time_)

        q_vector *= self.IDF_vector

        time_ = time()
        correlation_matrix = self.similarity(q_vector)
        print("created correlation matrix: ", time() - time_)
        time_ = time()
        correlation_matrix.sort(key=lambda x: -x[1])
        print("sorted correlation matrix: ", time() - time_)
        return correlation_matrix[:k]


if __name__ == '__main__':
    start = time()

    nlp = spacy.load("en_core_web_sm")
    BOOKS_PATH = "C:\\books\\"
    ps = PorterStemmer()
    saved_files = [join(BOOKS_PATH, f"{i}.txt") for i in range(1, 51727)]

    all_tokens_and_indexes = get_all_tokens_and_indexes()
    print(f"getting tokens and indexes time: {time() - start}")
    SE = SearchEngine(all_tokens_and_indexes)
    corr_matrix = SE.search(["what", "a", "coincidence"], k=3, similarity=2)
    print(f"total time: {time() - start}")

