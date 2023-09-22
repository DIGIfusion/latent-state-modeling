from enum import Enum 

class D(Enum): 
    dynamic = 0
    slice = 1

class M(Enum): 
    vae = 0
    linear = 1
    DIVA = 2
    ssvae = 3
    vae_aux = 4