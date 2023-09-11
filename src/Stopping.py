import numpy as np
import numba
np.seterr(all='ignore')


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def inverse(E: np.ndarray, params: np.ndarray) -> np.ndarray:
    lnE = np.log(E)
    lnE2 = lnE * lnE
    lnE3 = lnE2 * lnE
    lnE4 = lnE3 * lnE
    return (params[0] +
            params[1] * lnE +
            params[2] * lnE2 +
            params[3] * lnE3 +
            params[4] * lnE4)


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def equation(E: np.ndarray, params: np.ndarray) -> np.ndarray:
    return 1/inverse(E, params)


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def inverseIntegral(E: np.ndarray, params: np.ndarray) -> np.ndarray:

    lnE = np.log(E)
    c1 = params[1] - 2.*params[2] + 6.*params[3] - 24.*params[4]
    return E*((params[0] - c1) + params[4] * lnE * lnE * lnE * lnE +
              (params[3] - 4.*params[4]) * lnE * lnE * lnE +
              (params[2] - 3.*params[3] + 12.*params[4]) * lnE * lnE +
              c1*lnE)


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def inverseIntegrate(E0: np.ndarray,
                     E1: np.ndarray,
                     params: np.ndarray) -> np.ndarray:
    """Inverse integral of stopping power calculates range of ion"""
    return inverseIntegral(E0, params) - inverseIntegral(E1, params)


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def inverseDiff(E: np.ndarray, params: np.ndarray) -> np.ndarray:

    lnE = np.log(E, dtype=np.float64)
    lnE2 = lnE * lnE
    lnE3 = lnE2 * lnE
    return (params[1] +
            2.*params[2]*lnE +
            3.*params[3]*lnE2 +
            4.*params[4]*lnE3)/E


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def EnergyAfterStopping(E0: np.ndarray,
                        X: np.ndarray,
                        params: np.ndarray,
                        E_THRESHOLD) -> np.ndarray:

    INT_from = inverseIntegral(E0, params)
    Eend = np.zeros_like(E0)
    isEnd = False
    _Eend = 0.0

    for i in numba.prange(0, E0.size):

        _Eend = E0[i]
        if isEnd:
            break

        for j in range(100):
            if _Eend < E_THRESHOLD:
                Eend[i] = 0
                isEnd = True
                break

            val = INT_from[i] - inverseIntegral(_Eend, params) - X[i]
            if np.abs(val) <= 0.02:
                Eend[i] = _Eend
                break

            _Eend = _Eend + val/inverse(_Eend, params)
    return Eend
