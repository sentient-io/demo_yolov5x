from ctypes import *
import numpy as np
import numpy.ctypeslib as nc
import os
import platform

os_name = platform.system()
if os_name == 'Linux':
    path = '../lib/libSoyNet.so'
elif os_name == 'Windows':
    path = '../lib/SoyNet.dll'

if os.path.exists(path):
    lib = cdll.LoadLibrary(path)
else:
    print("Can't find SoyNet Library")
    exit(-1)

lib.initSoyNet.argtypes = [c_char_p, c_char_p]
lib.initSoyNet.restype = c_void_p
def initSoyNet(cfg, extent_params="") :
    if extent_params is None : extent_params=""
    return lib.initSoyNet(cfg.encode("utf8"), extent_params.encode("utf8"))

lib.feedData.argtypes=[c_void_p, c_void_p]
lib.feedData.restype=None
def feedData(handle, data) :
    lib.feedData(handle, data.ctypes.data_as(c_void_p))

lib.inference.argtypes=[c_void_p]
lib.inference.restype=None
def inference(handle) :
    lib.inference(handle)


class RIP(Structure):
    _fields_ = [("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float), ("id", c_int),
                ("prob", c_float)]


lib.getOutput.argtypes=[c_void_p, c_void_p]
lib.getOutput.restype=None
def getOutput(handle, outputs) :
    lib.getOutput(handle, outputs.ctypes.data_as(c_void_p)) #

lib.freeSoyNet.argtypes=[c_void_p]
lib.freeSoyNet.restype=None
def freeSoyNet(handle) :
    lib.freeSoyNet(handle)
