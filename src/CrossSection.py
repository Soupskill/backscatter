import os
import numpy as np
from typing import Callable
from mendeleev import element
from src.utilsRBS import Rutherford
from src.Files import R33
from src.Globals import CS_FOLDER


class CrossSection:
    """
    CrossSection a class that contains numric information about
    requred cross-section in mb/sr

    Example: elastic backscattering of 1H on 28Si
    
    crosssection = CrossSection(Isotope('28Si'), Beam('1H'), Beam('1H'))
    Target isotope _______________________^        ^            ^
    Incident beam _________________________________|            |
    outgoing beam_______________________________________________|
    
    by defaul rutherford cross-section scattering is used
    to calculate numeric values from energy you should use CrossSection.calculate(Energy)
    to set own values of cross-section to interpolate you should use CrossSection.updateByArray(Energy, Values)
    to choose cross-section from .r33 file CrossSection.selectR33(fname) 
    """
    __slots__ = ('theta',
                 'motherAtom',
                 'beamIn',
                 'beamOut',
                 'dautherAtom',
                 '__function',
                 'r33')

    def __init__(self,
                 motherAtom,
                 beamIn,
                 beamOut,
                 theta: float) -> None:
        
        from src.Element import Beam, VIsotope
        self.r33: R33 = None
        self.theta: float = theta
        self.motherAtom: VIsotope = motherAtom
        self.beamIn: Beam = beamIn
        self.beamOut: Beam = beamOut

        self.dautherAtom = VIsotope(
            element(self.motherAtom.Z + self.beamIn.Z - self.beamOut.Z).symbol,
            self.motherAtom.Z + self.beamIn.Z - self.beamOut.Z,
            self.motherAtom.A + self.beamIn.A - self.beamOut.A,
        )

        self.__function: Callable[[np.ndarray], np.ndarray] = None

    def __repr__(self) -> str:
        M = f'{round(self.motherAtom.A)}{self.motherAtom.symbol}'
        beamIn = f'{round(self.beamIn.A)}{self.beamIn.symbol}'
        beamOut = f'{round(self.beamOut.A)}{self.beamOut.symbol}'
        D = f'{round(self.dautherAtom.A)}{self.dautherAtom.symbol}'
        return f'{M}({beamIn},{beamOut}){D}_{int(self.theta)}'

    def calculate(self, E: np.ndarray) -> np.ndarray:
        """
        E is sorted array of energies in keV
        interpolated values of cross-section
        """
        return self.__function(E)

    def updateByArray(self, E: np.ndarray, sigma: np.ndarray) -> None:
        """
        E is sorted array of energies in keV
        sigma is value of differential cross-section in mb
        """
        self.__function = lambda E_to_interp: np.float32(
                            np.interp(E_to_interp,
                                        E,
                                        sigma,
                                        left=0,
                                        right=0))

    def selectR33(self, fname: str = None) -> None:
        """
        Select experimental cross-section from R33 file
        """
        if fname is None:
            fname = f'{str(self)}.r33'
        else:
            fname = f'{fname}.r33'

        if not os.path.isfile(CS_FOLDER + fname):
            self.selectRutherford()
            return

        self.r33 = R33(CS_FOLDER, fname)

        if self.r33.units == 'mb':

            self.__function = lambda E: np.float32(
                np.interp(E,
                          self.r33.data[:, 0],
                          self.r33.data[:, 1] * 1e-27,
                          left=0, right=0)
                          )

        elif self.r33.units == 'rr':

            self.__function = lambda E: np.float32(
                np.interp(E, self.r33.data[:, 0],
                            self.r33.data[:, 1] * Rutherford(
                                                    self.r33.data[:, 0],
                                                    self.beamIn.Z,
                                                    self.motherAtom.Z,
                                                    self.beamIn.A,
                                                    self.motherAtom.A,
                                                    self.theta) * 1e-27,
                                                    left=0, right=0)
                        )

    def selectRutherford(self) -> None:

        self.__function = lambda E: Rutherford(E,
                                                self.beamIn.Z,
                                                self.motherAtom.Z,
                                                self.beamIn.A,
                                                self.motherAtom.A,
                                                self.theta) * 1e-27
