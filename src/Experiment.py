import glob
import re
import numpy as np
from threading import Thread
from typing import List
from .Files import MPAFile


class Spectrum:

    def __init__(self,
                 energy: float,
                 charge: float,
                 ADCchannels: np.ndarray):

        self.energy = energy
        self.charge = charge
        self.ADCchannels = ADCchannels


class Measurement:

    def __init__(self,
                 angle: float,
                 path: str,
                 detectorId: int,
                 detectorLinear: float,
                 detectorOffset: float,
                 solidAngle: float) -> None:
        
        self.angle = angle
        self.path = path
        self.detectorId = detectorId
        self.detectorLinear = detectorLinear
        self.detectorOffset = detectorOffset
        self.solidAngle = solidAngle
        self.spectra: List[Spectrum] = []

        tread = Thread(target=self.loadspectra)
        tread.start()
        tread.join()
        
    def loadspectra(self):
        self.spectra = []
        
        for fname in glob.glob(f'{self.path}*.mpa'):
            self.spectra.append(self._loadspectrum(fname))

    def _loadspectrum(self, fname):
        
        _fname = fname.split('\\')[-1].split('/')[-1]
        energy = re.findall(r'\d{3,4}', _fname)
        charge = re.findall(r'Q=\d{1,2}',  _fname)
        return Spectrum(
            energy=float(energy[0]),
            charge=float(charge[0].replace('Q=', '')),
            ADCchannels=MPAFile('', fname).data[self.detectorId][:, 1])
