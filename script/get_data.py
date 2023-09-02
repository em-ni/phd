import Sofa
import Sofa.Gui


def main():

    root = Sofa.Simulation.getNode('root')


    # Initialization of the scene will be done here
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()

    # Accessing and printing the final time of simulation
    # "time" being the name of a Data available in all Nodes
    finalTime = root.time.value
    print(finalTime)

if __name__ == "__main__":
    main()