#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Sofa
import Sofa.Core
from Sofa.constants import *

class ContactListener(Sofa.Core.Controller):
    def __init__(self, *a, **kw): # arga, kwargs
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.root_node = kw['node']
        self.object_name = kw['object_name']
        self.collision_name = kw['collision_name']

    def onEvent(self, event):
        #print("ContactListener: onEvent")
        #print(event)

        # Access object_name colllision model
        object = self.root_node.getChild(self.object_name).getChild(self.collision_name)
        if object == None:
            print("ContactListener: object not found")
            return
        mech = object.MechanicalObject
        if mech == None:
            print("ContactListener: MechanicalObject not found")
            return
        else:
            print("ContactListener: MechanicalObject found")
            print("Forces: ", mech.force.value)
            print("External Forces: ", mech.externalForce.value)
            # Print force and external force values
            
