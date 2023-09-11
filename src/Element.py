from typing import List
from mendeleev import element
from src.CrossSection import CrossSection
from src.Globals import ABUNDANCE_THRESHOLD


class VIsotope:

    __slots__ = ('symbol', 'A', 'Z')
    
    def __init__(self, symbol: str, Z: int, A: float) -> None:
        self.symbol: str = symbol
        self.Z: int = Z
        self.A: int = A


class Beam(VIsotope):

    __slots__ = ('_energy', '_energySpread')
    def __init__(self, symbol: str, Z: int, A: float) -> None:
        super().__init__(symbol, Z, A)
        self._energy: float = 2000.
        self._energySpread: float = 3.

    @property
    def Energy(self): return self._energy

    @property
    def EnergySpread(self): return self._energySpread

    @Energy.setter
    def EnergySpread(self, E: float):
        self._energySpread = E

    @Energy.setter
    def Energy(self, E: float):
        self._energy = E    


class Isotope(VIsotope):

    __slots__ = ('abundance', 'crossSection')

    def __init__(self,
                 symbol: str,
                 Z: int,
                 A: float,
                 abundance: float) -> None:

        super().__init__(symbol, Z, A)
        self.abundance: float = abundance
        self.crossSection: CrossSection = None

    def getCross_SectionInstance(self,
                                 beamIn: Beam,
                                 beamOut: Beam,
                                 theta: float) -> None:

        self.crossSection = CrossSection(self, beamIn, beamOut, theta)

    def __repr__(self) -> str:

        return f'{round(self.A)}{self.symbol}'


class Element:

    __slots__ = ('symbol', 'isotopes', 'Z')

    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol
        self.isotopes: List[Isotope] = []
        _element = element(self.symbol)
        self.Z = _element.atomic_number

        for isotope in _element.isotopes:
            if isotope.abundance:
                if isotope.abundance > ABUNDANCE_THRESHOLD*100:
                    self.isotopes.append(

                        Isotope(self.symbol,
                                _element.atomic_number,
                                isotope.mass,
                                isotope.abundance/100)
                    )
    
    def __repr__(self) -> str:
        
        return f'{self.symbol}'
