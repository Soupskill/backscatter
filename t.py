import numpy as np
import time
import logging
from functools import wraps
import os
from threading import Thread, get_ident
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock




def psave(func):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="logger.log",
                        filemode='a',
                        level=logging.DEBUG,
                        format="%(asctime)s: [%(level)s] %(message)s")
    lock = Lock()

    @wraps(func)
    def inner(*args, **kwargs):
        lock.acquire()
        func(*args)
        lock.release()
    return inner


@psave
def pprint(msg):
    logging.info(msg)

def slave(index):
    
    pprint(f"hello from {os.getpid()}") 
    
    return index * 2


def main():

    with ProcessPoolExecutor(4) as pool:
        return list(pool.map(slave,[i for i in range(21,0, -1)]))

if __name__=="__main__":

    print(main())