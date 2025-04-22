#ifndef USB1408FS_HPP
#define USB1408FS_HPP

#include <libusb-1.0/libusb.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string>

class USB1408FS {
public:
    USB1408FS(const std::string& serverIP, int serverPort);
    ~USB1408FS();
    bool findDevice();
    void initialize();
    void configurePorts();
    void readInputs();
    bool initTCPConnection();
    void sendData(const std::string& data);

private:
    libusb_device_handle *udev;
    bool deviceFound;
    int sockfd;
    std::string serverIP;
    int serverPort;

    void checkLibUSBError(int result, const char* context);
    float convertToVoltage(int digitalValue, uint8_t gain) const;
    float convertToPsi(float voltage) const;
};

#endif // USB1408FS_HPP