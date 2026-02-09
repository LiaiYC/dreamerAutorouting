# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:32:58 2025

@author: user
"""

from . import circuit_routing

try:
    circuit_routing.register()
except:
    pass