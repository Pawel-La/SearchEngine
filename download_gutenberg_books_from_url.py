from concurrent.futures import ThreadPoolExecutor
from os.path import join
from urllib.request import urlopen

BOOKS_PATH = "C:\\Users\\Paweł\\Documents\\mownit\\books\\"
BOOK_ALREADY_DOWNLOADED = 1
TOTAL_NUMBER_OF_BOOKS = 70000


def download_from_url(url_path):
    try:
        with urlopen(url_path, timeout=3) as connection:
            return connection.read()
    except FileNotFoundError or FileExistsError:
        return None


def download_book(book_id):
    url1 = f"https://gutenberg.org/files/{book_id}/{book_id}-0.txt"
    url2 = f"https://gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    data = download_from_url(url1)
    if data:
        save_file = join(BOOKS_PATH, f"{book_id}.txt")
        with open(save_file, 'wb') as file:
            file.write(data)
    else:
        data = download_from_url(url2)
        if data:
            save_file = join(BOOKS_PATH, f"{book_id}.txt")
            with open(save_file, 'wb') as file:
                file.write(data)


def download_books():
    with ThreadPoolExecutor() as exe:
        _ = [exe.submit(download_book, book_id) for book_id in
             range(BOOK_ALREADY_DOWNLOADED, TOTAL_NUMBER_OF_BOOKS)]
