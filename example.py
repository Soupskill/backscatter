#! python3
import numpy as np
import time
np.float_ = np.float32
from matplotlib import pyplot as plt
from src.Detector import Detector
from src.Element import Beam
from src.Geometry import Geometry
from src.Simulation import Simulation
from queue import Queue
from threading import Thread

queue = Queue(10)

def createSimulation() -> Simulation:

    """
    To setup model experiment (Simulation) you must set detector,
    incident beam and geometry
    """
    detectorExample = Detector(linear=4.06356963, offset=74, solidAngle=1.7e-3, resolution=25., Nchannels=2048)
    beamExample = Beam(symbol='He', Z=2, A=4.0026)
    # Set energy of beam. default value is 2000 keV
    beamExample.Energy = 4483 
    # Set energy resolution of beam
    beamExample.EnergySpread = 1.
    
    # geometry class indicates all angles target and foil composition  
    geometryExample = Geometry()
    geometryExample.alpha = 0.
    geometryExample.theta = 170
    """
    less number of the target layer means that
    the layer closer to particle source
    
    """
    geometryExample.addLayerToTarget('asf', 1e5)
    geometryExample.target[0].addElement('C', 1.)
    """less number of the foil layer means that
    the layer closer to target
    """
    geometryExample.addLayerToFoils('dead', 250)
    geometryExample.foils[0].addElement('Si', 1)
    simulation = Simulation(beamExample, geometryExample, detectorExample)
    
    # number of incident particles/steradian state directly
    simulation.Qsr = 2.11e11

    return simulation

def worker():

    simulation = createSimulation()
    simulation.initSpectra()
    res = simulation.run()
    plt.plot(res['12C'])
    plt.show()
    
if __name__ == "__main__":

    
    worker()
