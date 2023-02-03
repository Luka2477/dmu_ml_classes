def print_list(list):
    for i, el in enumerate(list):
        print(f"{i} : {el}")
    print()


if __name__ == '__main__':
    books = ["Harry Potter", "Lord Of The Rings", "Kafka on the Shore", "Norwegian Wood", "The Great Gatsby"]
    print_list(books)

    books.append("To Kill A Mockingbird")
    print_list(books)

    books.pop(2)
    print_list(books)

    booksLen = len(books)
    print(f"Books list length is {booksLen}")

    booksReverse = books.reverse()
    print_list(books)

