import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod


class VFile(ABC):

    def __init__(self, path: str, fname: str) -> None:
        self.fname = path + fname
        self.buffLines: List[str] = None
        self._read()

    def _read(self) -> None:
        with open(self.fname, 'rb') as _file:
            self.buffLines = _file.read().decode('latin1').split('\n')

    @abstractmethod
    def parse(self):
        pass


class R33(VFile):

    def __init__(self, path, fname):

        super().__init__(path, fname)
        self.units: str = None
        self.reaction: str = None
        self.authors: str = None
        self.source: str = None
        self.data: np.ndarray = np.array(())
        self.parse()

    def parse(self):

        bounds = []
        for i, line in enumerate(self.buffLines):
            if 'Source:' in line:
                self.source = line.split(':')[1].strip()
            if 'Name:' in line:
                self.authors = line.split(':')[1].strip()
            if 'Units:' in line:
                self.units = line.split(':')[1].strip()
            if 'Reaction:' in line:
                self.reaction = line.split(':')[1].strip()
            if 'Data:' in line:
                bounds.append(i)

        self.data = np.array(())
        sep = ',' if ',' in self.buffLines[1] else ' '
        for i in range(bounds[0] + 1, bounds[1] - 2):

            self.data = np.append(self.data,
                                  np.fromstring(self.buffLines[i], sep=sep))

        dim = np.fromstring(self.buffLines[i], sep=sep).size
        self.data.resize(len(self.data)//dim, dim)
        if (dim > 2): self.data = self.data[:, ::2]


class MPAFile(VFile):

    def __init__(self, path, fname):
        super().__init__(path, fname)
        self.data: Dict[int, np.ndarray] = {}
        self.parse()

    def parse(self):
        """MPA file can contains multiple ADC channels
            id corresponds to number of detector """

        ranges = {}
        startData = []

        for i, line in enumerate(self.buffLines):

            if '[DATA0' in line:
                ranges[0] = int(line.split(',')[1].replace(']', '').strip())
            if '[DATA1' in line:
                ranges[1] = int(line.split(',')[1].replace(']', '').strip())
            if '[DATA]' in line:
                startData.append(i+1)

        for i, index in enumerate(startData):

            tmp = np.array((0, 0))
            for line in self.buffLines[index:index + ranges[i]]:

                _line = line.strip()
                n = _line.find(' ')
                if n != -1:
                    _line = _line[:n] + '\t' + _line[n+1:]
                _line = _line.replace(' ', '')
                _line = _line.split('\t')

                tmp = np.append(tmp, tuple(map(int, _line)))
            self.data[i] = tmp.reshape((len(tmp)//2, 2))
