import numpy as np
import plumedCommunications

plumedInit={"Value": plumedCommunications.defaults.COMPONENT,}

def myUpdate(_: plumedCommunications.PythonCVInterface):
    print("myUpdate")
    return {}

def plumedCalculate(action: plumedCommunications.PythonCVInterface):
    at: np.ndarray = action.getPositions()
    return at[0][0]+at[0][1]
