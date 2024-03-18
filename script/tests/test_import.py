# Description: test the import of python modules
# Possible errors: 
# python version, SOFA version, SOFA plugins version, SOFA python version
# SOFA plugins path, SOFA python path
# plugins not built with -DPLUGIN_XXX=ON

# ./build/v22.12/lib/python3/site-packages/
import Sofa
print("Sofa imported from ", Sofa.__file__)

import Sofa.Components
print("Sofa.Components imported from ", Sofa.Components.__file__)

import Sofa.Core
print("Sofa.Core imported from ", Sofa.Core.__file__)

import Sofa.CosseratPlugin
print("Sofa.CosseratPlugin imported from: ", Sofa.CosseratPlugin.__file__)

import softrobots
print("softrobots imported from: ", softrobots.__file__)

import matplotlib
print("matplotlib imported from: ", matplotlib.__file__)
