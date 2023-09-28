from typing import List, Union
from src.Layer import Layer


class Geometry:

    def __init__(self):

        self._theta: Union[float, int] = 170
        self._alpha: Union[float, int] = 10
        self._beta: Union[float, int] = 0
        self.target: List[Layer] = []
        self.foils: List[Layer] = []
        self.windows: List[Layer] = []
        self._targetcount: int = 0
        self._foilcount: int = 0

    @property
    def alpha(self): return self._alpha

    @property
    def beta(self): return self._beta

    @property
    def theta(self): return self._theta

    @theta.setter
    def theta(self, theta):

        self._theta = theta
        self._beta = 180. - self._theta - self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._beta = 180. - self._theta - self._alpha

    def addLayerToTarget(self, name: str, thickness: float):
        self.target.append(Layer(name, thickness, self._targetcount))
        self._targetcount += 1

    def addLayerToFoils(self, name: str, thickness: float):
        self.foils.append(Layer(name, thickness, self._foilcount))
        self._foilcount += 1
    
    def __repr__(self) -> str:
        
        return f'Geometry:\n\talpha {self.alpha}\n\ttheta {self.theta}\n\ttarget {[target.name for target in self.target]}'
