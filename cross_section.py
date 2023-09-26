#! python3
import numpy as np
import os
import re
import glob
import logging
from typing import List
from scipy.optimize import minimize
np.float_ = np.float32
from matplotlib import pyplot as plt
from src.Detector import Detector
from src.Files import MPAFile
from src.Element import Beam
from src.Geometry import Geometry
from src.Simulation import Simulation
from concurrent.futures import ProcessPoolExecutor

THETA = 170
DETID = 0
ENERGY = np.arange(3500, 6500, 5)

class Spectrum:
    
    def __init__(self, energy: float, data: np.ndarray):
        self.energy = energy
        self.data = data

def applyAcceleratorCalibration(energy: float, 
                                linear=1., 
                                offset=0., 
                                charge_state=2, 
                                extraction_v=20., ):
    return ((energy - extraction_v) / (charge_state + 1) * linear - offset) *\
            (charge_state + 1) + extraction_v

def loadSpectrum(fname) -> Spectrum:
    mpa = MPAFile('', fname)
    energy = float(re.findall(r'E=\d{4}', fname)[0].strip('E='))
    energy = applyAcceleratorCalibration(energy, 0.9975, -1.4)
    return Spectrum(energy, mpa.data[DETID][:,1]) 
    

def collectSpectra(path: str) -> List[Spectrum]:

    if not os.path.isdir(path): return
    with ProcessPoolExecutor(os.cpu_count()) as pool:        
        return list(pool.map(loadSpectrum, glob.glob(f'{path}*.mpa')))


def createSimulations(spectra: List[Spectrum]) -> List[Simulation]:


    """
    To setup model experiment (Simulation) you must set detector,
    incident beam and geometry
    """
    simulations = []
    for spectrum in spectra:
            
        detectorExample = Detector(linear=4.06356963, offset=74, solidAngle=1.7e-3, resolution=25., Nchannels=2048)
        beamExample = Beam(symbol='He', Z=2, A=4.0026)
        # Set energy of beam. default value is 2000 keV
        beamExample.Energy = spectrum.energy 
        # Set energy resolution of beam
        beamExample.EnergySpread = 1.
        
        # geometry class indicates all angles target and foil composition  
        geometryExample = Geometry()
        geometryExample.alpha = 0.
        geometryExample.theta = THETA
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
        simulations.append(simulation)
    
    return simulations

def calculateChi2(simulation: Simulation, spectrum: Spectrum) -> float:

    theor_spectrum = np.zeros_like(simulation.detector.EnergyChannels)
    res = simulation.run()
    for isotope in res.keys():
        theor_spectrum += res[isotope]
    
    thres = np.where(theor_spectrum > 300)[0][-1]
    return np.sum((theor_spectrum[thres - 10: thres - 40] - spectrum.data[thres - 10: thres - 40]) ** 2)
    

def getTotalChi2(simulations: List[Simulation], spectra: List[Spectrum]) -> float:

    contex = zip(simulations, spectra)
    with ProcessPoolExecutor(os.cpu_count()) as processPool:

        chi2 = np.sum(processPool.map(calculateChi2, contex))
    
    return chi2


def update_task(cross_section, simulations: List[Simulation], spectra: List[Spectrum]):

    for simulation in simulations:
        simulation.geometry.target[0].getIsotopeByName('12C').crossSection.updateByArray(
            ENERGY,
            cross_section
        )
        simulation.initCrossSection()
    return getTotalChi2(simulations, spectra)    
    



def main():

    spectra = collectSpectra("")
    simulations = createSimulations(spectra)
    cross_section = simulations[0].geometry.target[0].getIsotopeByName('12C').crossSection.calculate(ENERGY)

    result = minimize(update_task, 
                        x0=cross_section, 
                        args=(simulations, spectra,), 
                        method="COBYLA", 
                        options={"rhobeg":1, "maxiter":1000})

    np.savetxt('cross_section.result', result['x'])