import Sofa
import Sofa.Core as SC

def main():
    import SofaRuntime
    import Sofa.Gui

    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("SofaImplicitOdeSolver")
    SofaRuntime.importPlugin("SofaLoader")
    SofaRuntime.importPlugin("SofaRigid")

    root = Sofa.Core.Node("root")
    Sofa.Simulation.init(root)
   
    Sofa.Gui.GUIManager.Init("load_data.scn", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1920, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()



#Run the script on the Python IDLE
if __name__ == "__main__":
    main()