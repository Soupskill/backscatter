import numpy as np
from src.utilsRBS import gauss


class Detector:

    __slots__ = ('_linear',
                 '_offset',
                 '_Nchannels',
                 '_resolution',
                 'solidAngle',
                 'resolution_ch',
                 'EnergyChannels',
                 'responce')

    def __init__(self,
                linear: float,
                offset: float,
                solidAngle: float,
                resolution: float,
                Nchannels: int):

        self._linear: float = linear
        self._offset: float = offset
        self._Nchannels: int = Nchannels
        self._resolution: float = resolution
        self.resolution_ch: float = None
        self._updateEnergyChannels()
        self.solidAngle: float = solidAngle

    @property
    def linear(self): return self._linear

    @linear.setter
    def linear(self, linear):
        self._linear = linear
        self._updateEnergyChannels()

    @property
    def offset(self): return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset
        self._updateEnergyChannels()

    @property
    def resolution(self): return self._resolution

    @property
    def resolutionCh(self): return int(self._resolution/self._linear)

    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution
        self._updateEnergyChannels()

    @property
    def Nchannels(self): return self._Nchannels

    @Nchannels.setter
    def Nchannels(self, N):
        self._Nchannels = N
        self._updateEnergyChannels()

    def _updateEnergyChannels(self):
        self.resolution_ch = self.resolution/self.linear/2.355
        self.EnergyChannels: np.ndarray = np.arange(self.Nchannels) \
                                        * self.linear + self.offset
        tmp = int(self.resolution_ch * 14)
        self.responce = gauss(np.arange(tmp+1),
                                1, tmp/2,
                                self.resolution_ch, 0)
        self.responce = self.responce/np.sum(self.responce)
