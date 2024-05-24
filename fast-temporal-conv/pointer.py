buff_len = 5
pointer = 0

for i in range(100):
    print(pointer)
    if pointer < buff_len - 1:
        pointer += 1
    else:
        pointer = 0
