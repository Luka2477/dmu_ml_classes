import math
import tkinter as tk


def is_prime(n):
    prime_flag = 0

    if n <= 1:
        return False

    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            prime_flag = 1
            break

    return prime_flag == 0


def number_sv_callback(sv, lbl):
    if sv.get() == "":
        return True

    sv_int = int(sv.get())
    is_prime_result = is_prime(sv_int)

    lbl.config(text="Number is a prime!" if is_prime_result else "Number is not a prime...")
    return True


if __name__ == '__main__':
    window = tk.Tk()

    tk.Label(window, text="Is the number a prime number???") \
        .grid(column=0, row=0, columnspan=2)

    tk.Label(window, text="Number: ") \
        .grid(column=0, row=1)

    result = tk.Label(window, text="Please enter a number :)")
    result.grid(column=0, row=2, columnspan=2)

    number_sv = tk.StringVar()
    number = tk.Entry(window, width=5, textvariable=number_sv, validate="all",
                      validatecommand=lambda: number_sv_callback(number_sv, result))
    number.grid(column=1, row=1)

    window.mainloop()
