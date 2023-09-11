import numpy as np
import time
from functools import wraps
import numba


def njit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        args = map(np.float32, args)
        for i in kwargs.keys():
            kwargs[i] = np.float32(kwargs[i])
        result = func(*args, **kwargs)
        return result
    return inner


@njit
def vector_shift(vector: np.ndarray, step: int) -> np.ndarray:
    """shift vector values to right by step"""
    new_vector = np.empty_like(vector)
    step = step % len(vector)
    if step == 0:
        return vector
    if step < 0:
        step = step % len(vector)
        step = len(vector) - step
    new_vector[step:] = vector[:-step]
    new_vector[:step] = vector[-step:]
    return new_vector


@njit
def bohr_spread(X: float, z1: float, z2: float) -> float:

    """Bohr`s straggling theory\nclear gaussian\nsigma^2 ~ z1^2*z2*X"""
    return 0.26 * z1 ** 2 * z2 * X * 1e-3


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def get_spread_responce(E: np.ndarray,
                        spreading: np.ndarray,
                        k: float) -> np.ndarray:

    matrix = np.zeros((E.size, E.size), dtype=np.float32)
    matrix[0, 0] = 1
    spreading += spreading * k * k
    spreading = np.sqrt(spreading)

    for i in numba.prange(1, E.size, 1):
        matrix[i, :] = gauss(E, 1, E[i], spreading[i], 0)
        matrix[i, :] /= np.sum(matrix[i, :])

    return matrix


@njit
def get_responce(size: int, resolution: float, linear: float) -> np.ndarray:
    """Get responce matrix for SSD detector"""
    matrix = np.empty((size, size))
    E = np.arange(0, size)
    tmp = gauss(E,
                1,
                size // 2,
                resolution / linear / 2 / np.sqrt(2 * np.log(2)), 0)
    tmp[np.where(tmp < 0.001)] = 0
    tmp /= np.sum(tmp)
    for i in range(size):
        matrix[i, :] = vector_shift(tmp, i + size // 2)
    matrix[size-10: size, 0: 20] = 0
    matrix[0: 20, size - 10: size] = 0
    return matrix


@numba.njit(nogil=True, fastmath=True, cache=True)
def gauss(x: np.ndarray(None, np.float32),
          a: np.float32,
          b: np.float32,
          c: np.float32,
          d: np.float32) -> np.ndarray(None, np.float32):
    """"y(x) = a*e^(-(x-b)^2/2c^2) + d"""
    return a * np.exp(-0.5 * (x - b) * (x - b) / c / c) + d


@njit
def smooth(array, window=5):
    filt = np.ones(window) / window
    return np.convolve(array, filt, mode='same')


@njit
def kinFactor(m: float, M: float, theta: float) -> float:
    """
    m - mass of projectile in aem
    M - mass of targer in aem
    theta - scattering angle in degrees
    """
    return 1/(1+M/m)**2*(np.cos(np.deg2rad(theta)) +
                         np.sqrt((M/m)**2 - np.sin(np.deg2rad(theta))**2))**2


@njit
def root(func, a, b, eps=2e-2) -> float:
    """
    find root of equation by bisection
    """
    if func(a)*func(b) < 0:
        err = np.abs(a - b)
        x = (a + b)/2
        while err > eps:
            if func(a)*func(x) < 0:
                b = x
            elif func(b) * func(x) < 0:
                a = x
            x = (a + b)/2
            err = np.abs(a - b)
        return x
    else:
        return -1


@njit
def find_extream(array: np.ndarray, mode: str) -> float:

    modes = {'min': np.argmin, 'max': np.argmax}

    # array = savgol_filter(array, 17, 1)
    mask = np.where(array > 500)[0][10:-10]
    point = modes[mode](array[mask]) + mask[0]

    coef = np.polyfit(x=np.arange(-15, 15, dtype=int) + point,
                      y=array[np.arange(-15, 15, dtype=int) + point],
                      deg=3)
    roots = np.roots(coef[:-1]*np.arange(3, 0, -1))

    return roots[np.argmin(np.abs(roots-point))]


@njit
def Rutherford(E: np.ndarray,
               z1: int,
               z2: int,
               m1: float,
               m2: float,
               theta: float) -> np.ndarray:
        """
        E: keV return cross-section in mb/sr
        """
        cost = np.cos(np.deg2rad(theta))
        sint = np.sin(np.deg2rad(theta))
        c = 5.1837436e6
        D = (z1 * z2 / E) ** 2
        A = (m2 ** 2 - m1 ** 2 * sint ** 2) ** 0.5 + m2 * cost
        B = m2 * sint ** 4 * (m2 ** 2 - m1 ** 2 * sint ** 2) ** 0.5
        return c * D * A ** 2 / B


@njit
def calc_chi2(arr1: np.ndarray, arr2: np.ndarray) -> float:

    nonZeroInd = np.where(arr1 != 0)
    tmp = (arr1[nonZeroInd] - arr2[nonZeroInd])**2/arr1[nonZeroInd]/len(arr1[nonZeroInd])

    return np.sum(tmp)


@njit
def applyEnergyCalibration(E0: float,
                           GVM_linear: float,
                           GVM_offset: float,
                           extraction_voltage: float,
                           ionCharge: float) -> float:

    return ((E0 - extraction_voltage)/(ionCharge+1) *
            GVM_linear + GVM_offset) * (ionCharge+1) + extraction_voltage


def minimize_(func, x0, step, niter, args):

    x = x0
    bestfun = func(x, *args)
    bestx = x

    cs = bestx
    dt = np.array((0, 0, 0))
    k = 0
    for i in range(niter):

        t = time.time()
        eps = np.random.uniform(1 - step, 1 + step, size=len(x))
        cs = bestx * eps
        value = func(cs, *args)

        if (bestfun > value):

            bestfun = value
            bestx = cs

        dt[k] = time.time() - t
        if k == 2:
            k = 0
        else:
            k += 1
        print(f'iter {i} of {niter}, func={bestfun: .2f}, EOT={np.average(dt) * (niter - i): .2f}', end='\r', flush=True)
    return {'x': bestx, 'fun': bestfun}
