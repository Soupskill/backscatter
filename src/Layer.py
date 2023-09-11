import os
import numpy as np
from typing import Dict, Callable
from src.Stopping import equation
from src.CrossSection import CrossSection
from src.Element import Element, Beam, Isotope
from src.Globals import STOPPING_FOLDER, SIGMACALC


class Layer:

    def __init__(self, name: str, thickness: float, index: int):

        self.__components: Dict[Element, float] = {}
        self.name = name
        self.stoppingParams = np.zeros(5)
        self.thickness = thickness
        self.index = index

    def addElement(self, symbol: str, fraction: float):
        self.__components[Element(symbol)] = fraction

    def removeElement(self, symbol: str):
        for element in self.__components.keys():
            if element.symbol == symbol:
                self.__components.pop(element)

    def setCrossSection(self, beamIn: Beam, beamOut: Beam, theta: float):

        if SIGMACALC:
            func: Callable[[CrossSection], None] = lambda CrossSection: CrossSection.selectR33()

        else:
            func = lambda CrossSection: CrossSection.selectRutherford()

        for element in self.__components.keys():
            for isotope in element.isotopes:
                isotope.getCross_SectionInstance(beamIn, beamOut, theta)
                func(isotope.crossSection)

    def checkFraction(self):
        tot = 0
        for fraction in self.__components.values():
            tot += fraction

        if np.abs(tot - 1) > 0.01:
            raise ValueError(f'Check fractions of layer elements {self.index}')
        print('Fraction check good!:)')

    def getComponents(self): return tuple(el for el in self.__components.items())

    def getIsotopeByName(self, name: str) -> Isotope:

        for el in self.__components.keys():
            for isotope in el.isotopes:
                if name == str(isotope):
                    return isotope

    def setBraggRule(self, beam: Beam):

        E = np.arange(200., 7000., 10., dtype=np.float32)
        stopping = np.zeros_like(E, dtype=np.float32)

        for element in self.__components.keys():

            fname = f'{STOPPING_FOLDER}{beam.Z}_{round(beam.A)}_{element.Z}.dat'
            if os.path.isfile(fname):
                params = np.loadtxt(fname)
            else:
                raise FileNotFoundError(f'File {fname} with stopping params for {element} not found')
            stopping += equation(E, params) * self.__components[element]

        lnE = np.log(E)
        Mat = lnE[:, np.newaxis]**[0, 1, 2, 3, 4]

        self.stoppingParams = np.linalg.lstsq(Mat, 1/stopping, rcond=-1)[0]
