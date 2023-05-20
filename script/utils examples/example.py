import Sofa.Core as SC
from SofaMAMMOBOT.nodes import catheter

# Create the root node
root = SC.Node("root")

# Load the SofaPython3 plugin
root.addObject('RequiredPlugin', name='SofaPython3')

# Catheter node with default values
catheter.CatheterNode(root)
