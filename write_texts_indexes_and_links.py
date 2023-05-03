from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor

old_texts_path = "C:\\old_texts.txt"
new_texts_path = "C:\\new_texts.txt"


def is_url_correct(url_path):
    try:
        with urlopen(url_path):
            return True
    except URLError or HTTPError:
        return False


def get_correct_url(url1, url2):
    if is_url_correct(url1):
        return url1
    if is_url_correct(url2):
        return url2
    return None


def write_index_and_url(book_id):
    url1 = f"https://gutenberg.org/files/{book_id}/{book_id}-0.txt"
    url2 = f"https://gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    correct_url = get_correct_url(url1, url2)
    if correct_url is None:
        print("error")
        return

    with FileLock(new_texts_path + ".lock"):
        with open(new_texts_path, 'a') as file:
            file.write(f"{book_id} {correct_url}\n")


def write_indexes_and_urls():
    with open(old_texts_path, "r", encoding='utf-8') as file:
        lines = file.readlines()
    with open(new_texts_path, 'w') as file:
        file.write("")

    texts = [int(line[:-1]) for line in lines]
    with ThreadPoolExecutor() as exe:
        _ = [exe.submit(write_index_and_url, idx) for idx in texts]
