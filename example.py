#! python3
import numpy as np
np.float_ = np.float32
from matplotlib import pyplot as plt
from src.Detector import Detector
from src.Element import Beam
from src.Geometry import Geometry
from src.Simulation import Simulation


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


def main():

    simulation = createSimulation()
    # zeroing partial spectra
    simulation.initSpectra()
    
    res = simulation.run()
    # keys in obtained result is str for example '12C', '56Fe'
    for isotope in res.keys():
        plt.plot(simulation.detector.EnergyChannels,
                 res[isotope],
                 label=isotope)

    plt.legend()
    plt.xlabel("E, keV")
    plt.ylabel("N")
    plt.xlim(0, 1400)
    plt.ylim(0, 20000)
    plt.show()


if __name__ == "__main__":

    main()
