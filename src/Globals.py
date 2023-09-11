import json

E_THRESHOLD = 200
CS_FOLDER = "./resources/cross_section/"
STOPPING_FOLDER = "./resources/stop/"
SIGMACALC = True 
MIN_YEILD_DISCRETIZATION = 256
ABUNDANCE_THRESHOLD = 0.01
STRAGGLING = 0
ENERGY_STEP = float(2.1)


def loadSettings(fname):

    with open(fname) as settingsFile:
        settings = json.load(settingsFile)
        global E_THRESHOLD, CS_FOLDER, STOPPING_FOLDER, SIGMACALC, MIN_YEILD_DISCRETIZATION, ABUNDANCE_THRESHOLD, STRAGGLING, ENERGY_STEP
        
        E_THRESHOLD = settings['E_THRESHOLD']
        CS_FOLDER = settings['CS_FOLDER']
        STOPPING_FOLDER = settings['STOPPING_FOLDER']
        SIGMACALC = settings['SIGMACALC']
        MIN_YEILD_DISCRETIZATION = settings['MIN_YEILD_DISCRETIZATION']
        ABUNDANCE_THRESHOLD = settings['ABUNDANCE_THRESHOLD']