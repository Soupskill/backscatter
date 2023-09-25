import numpy as np
import time
from threading import Thread, get_ident
from concurrent.futures import ThreadPoolExecutor

FIRST = "done"
SECOND = "exit"

def worker(i):

    a = np.random.uniform(0, 1, size=(5000))
    b = np.random.uniform(0, 1, size=(5000))
##    print(f'{FIRST} thread {get_ident()}, {i}')
    for i in range(1000):
        c = a * b + i
##    print(f'{SECOND} thread {get_ident()}, {i}')


t = time.time()
for i in range(100):
    worker(0)

print(time.time() - t)
##
##t = time.time()
##with ThreadPoolExecutor(10) as pool:
##
##    res = pool.map(worker, list(range(100)))
##print(time.time() - t)


