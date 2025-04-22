#include "USB1408FS.hpp"

int main(int argc, char **argv)
{
    std::string serverIP = "127.0.0.1";
    int serverPort = 5000;

    USB1408FS daqSystem(serverIP, serverPort);
    
    if (daqSystem.findDevice()) {
        daqSystem.initialize();
        daqSystem.configurePorts();
        while (true) {
            daqSystem.readInputs();
        }
    }
    return 0;
}
