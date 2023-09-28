#! python3
import numpy as np
import os
import sys
import re
import glob
import time
import logging
from typing import List
from scipy.optimize import minimize
from matplotlib import pyplot as plt
np.float_ = np.float32
from src.Detector import Detector
from src.Files import MPAFile
from src.Element import Beam
from src.Geometry import Geometry
from src.Simulation import Simulation
from src.utilsRBS import minimize_
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock, Event
from queue import Queue


logger = logging.getLogger(__name__)
hangler = logging.FileHandler('log.log')
formater = logging.Formatter("%(asctime)s : [%(levelname)s] %(message)s")
hangler.setFormatter(formater)
hangler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(hangler)


THETA = 170
SOLIDANGLE = 1.7e-3
DETID = 0

earea = [np.arange(3000, 3561, 50),
            np.arange(3562, 3615, 6),
            np.arange(3700, 4100, 60),
            np.arange(4150, 4350, 6),
            np.arange(4360, 5150, 30),
            np.arange(5220, 5320, 6),
            np.arange(5350, 5735, 50),]
ENERGY = np.concatenate(earea)


# ENERGY = np.arange(3000, 6500, 5)

queue_chi2 = Queue()
plotQueue = Queue(1)
lock = Lock()
new_cross_section = np.zeros_like(ENERGY)

class Spectrum:
    
    def __init__(self, energy: float, particlessr: float, data: np.ndarray):
        self.energy = energy
        self.particlessr = particlessr
        self.data = data
    
    def __repr__(self) -> str:
        return f'spectrum <E={self.energy:.2f}>'

def applyAcceleratorCalibration(energy: float, 
                                linear=1., 
                                offset=0., 
                                charge_state=2, 
                                extraction_v=20., ):
    return ((energy - extraction_v) / (charge_state + 1) * linear - offset) *\
            (charge_state + 1) + extraction_v

def loadSpectrum(fname) -> Spectrum:
    logger = logging.getLogger(__name__)
    try:
        mpa = MPAFile('', fname)
    except Exception as exc:
        logger.error(f'file {fname} cannot be parced with {exc}')
        return

    energy = float(re.findall(r'E=\d{4}', fname)[0].strip('E='))
    energy = applyAcceleratorCalibration(energy, 0.9975, -1.4)
    charge = float(re.findall(r'Q=\d{1,2}', fname)[0].strip('Q='))
    particlessr = charge * 1e-6 / 1.6e-19 * SOLIDANGLE
    spectrum = Spectrum(energy, particlessr, mpa.data[DETID][:,1]) 
    
    return  spectrum
    

def collectSpectra(path: str) -> List[Spectrum]:

    if not os.path.isdir(path): return
    with ThreadPoolExecutor(10) as pool:        
        res = list(pool.map(loadSpectrum, glob.glob(f'{path}*.mpa')))
    return res

def createSimulations(spectrum: Spectrum) -> Simulation:

    detectorExample = Detector(linear=4.06356963, offset=74, solidAngle=SOLIDANGLE, resolution=15., Nchannels=728)
    beamExample = Beam(symbol='He', Z=2, A=4.0026)
    beamExample.Energy = spectrum.energy 
    beamExample.EnergySpread = 1.
    geometryExample = Geometry()
    geometryExample.alpha = 0.
    geometryExample.theta = THETA
    geometryExample.addLayerToTarget('asf', 1e5)
    geometryExample.target[0].addElement('C', 1.)
    geometryExample.addLayerToFoils('dead', 250)
    geometryExample.foils[0].addElement('Si', 1)
    simulation = Simulation(beamExample, geometryExample, detectorExample)
    simulation.Qsr = spectrum.particlessr
    
    return simulation



def worker(spectrum: Spectrum, event: Event):
    with lock:
        simulation = createSimulations(spectrum)
    
    while 1:
        event.wait()
        with lock:
            simulation.geometry.target[0].getIsotopeByName('12C').crossSection.updateByArray(ENERGY, new_cross_section)
        simulation.initSpectra()
        v = calculateChi2(simulation, spectrum)
        queue_chi2.put(v)
        event.clear()

def calculateChi2(simulation: Simulation, spectrum: Spectrum) -> float:

    theor_spectrum = np.zeros_like(simulation.detector.EnergyChannels)
    res = simulation.run()
    for isotope in res.keys():
        theor_spectrum += res[isotope]
    thres = np.where(spectrum.data > 50)[0][-1]
    return np.sum((theor_spectrum[thres -40: thres - 7] - spectrum.data[thres - 40: thres - 7]) ** 2) / (40-7)


def update_task(cross_section, events: List[Event]):
    global new_cross_section
    with lock:
        new_cross_section = cross_section
    
    list(map(Event.set, events))
    chi2 = 0
    i = 0
    while not all([not evt.isSet() for evt in events]):
        pass
    while not queue_chi2.empty():
        chi2 += queue_chi2.get()
    chi2 = chi2 / len(events)
    print(chi2)
    return chi2


def minimizing_task(initial_cross_section, events):
    print('start')
    result = minimize_(update_task, x0=initial_cross_section, step=3e-25, niter=5000, args=(events,))


def main():
    logger = logging.getLogger(__name__)
    logger.info(f'app started')
    spectra = collectSpectra("C:/Users/CHE/Desktop/Тимофей/RBSSim/exitation/")[1:]            
    if None in spectra:
        logger.error(f'NO SPECTRA DATA. exit from program')
        return
    
    events: List[Event] = []
    threads: List[Thread] = []
    for spectrum in spectra:
        events.append(Event())
        threads.append(Thread(target=worker, args=(spectrum,events[-1]), daemon=True))
    
    queue_chi2.maxsize = len(threads)
    
    
    sim = createSimulations(spectra[0])
    
    initial_cross_section = sim.geometry.target[0].getIsotopeByName('12C').crossSection.calculate(ENERGY)
    # initial_cross_section = np.ones_like(ENERGY) * 5e-25
    list(map(Thread.start, threads))
    result = minimize(update_task, x0=initial_cross_section, args=(events,), method='COBYLA', options={'rhobeg': 0.3e-25, 'maxiter': 1000})
    # minimizing_task(initial_cross_section, events, )
    
    # while 1:
    #     try:
    #         data = plotQueue.get()

    #         plt.clf()
    #         plt.plot(ENERGY, initial_cross_section)
    #         plt.plot(ENERGY, data)
    #         plt.pause(0.1)
    #         plt.draw()
    #     except:
    #         pass
    
    
    # resultfilename = 'cross_section.result.txt'  
    # np.savetxt(resultfilename, result['x'])

    # logger.info(f'results in {resultfilename}')
    logger.info(f'app successfully closed')
    
if __name__ == "__main__":

    main()