import logging
import networkx as nx
import numpy as np
from functools import partial, partialmethod
import sys

#import community as community_louvain
# from .my_surgery import ARI

logging.TRACE = logging.DEBUG + 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)

logging.StreamHandler(stream=sys.stdout)
logger = logging.getLogger("GeometricNetworkAnalysis")
# logger = logging.getLogger("GNA")

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_verbose(verbose="INFO"): # "ERROR"):
    """ Set up the verbose level of GeometricNetworkAnalysis.
    
    Parameters  
    ----------   
    verbose: {"INFO","TRACE","DEBUG","ERROR"} 
        Verbose level. (Default = "ERROR")
            - "INFO": show only iteration process log. 
            - "TRACE": show detailed iteration process log.
            - "DEBUG": show all output logs. 
            - "ERROR": only show log if error happened. 
    """
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "TRACE":
        logger.setLevel(logging.TRACE)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print("Unrecognized verbose level, options: ['INFO','DEBUG','ERROR'], use 'ERROR' instead")
        logger.setLevel(logging.ERROR)       