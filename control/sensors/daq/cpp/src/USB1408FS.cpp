#include "USB1408FS.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <cstring>
#include <unistd.h> 

#include "pmd.h"
#include "usb-1408FS.h"

USB1408FS::USB1408FS(const std::string& serverIP, int serverPort) 
: udev(nullptr), deviceFound(false), sockfd(-1), serverIP(serverIP), serverPort(serverPort) {
    int result = libusb_init(NULL);
    checkLibUSBError(result, "Initialize libusb");
    if (!initTCPConnection()) {
        std::cerr << "Failed to initialize TCP connection\n";
        exit(EXIT_FAILURE);
    }
}

USB1408FS::~USB1408FS() {
    if (udev) {
        libusb_close(udev);
    }
    libusb_exit(NULL);
    if (sockfd != -1) {
        close(sockfd);
    }
}

bool USB1408FS::findDevice() {
    udev = usb_device_find_USB_MCC(USB1408FS_PID, NULL);
    deviceFound = (udev != nullptr);
    if (deviceFound) {
        std::cout << "USB-1408FS Device is found!\n";
    } else {
        std::cout << "No device found.\n";
    }
    return deviceFound;
}

void USB1408FS::initialize() {
    if (deviceFound) {
        init_USB1408FS(udev);
    }
}

void USB1408FS::configurePorts() {
    usbDConfigPort_USB1408FS(udev, DIO_PORTA, DIO_DIR_OUT);
    usbDConfigPort_USB1408FS(udev, DIO_PORTB, DIO_DIR_IN);
    usbDOut_USB1408FS(udev, DIO_PORTA, 0);
}

void USB1408FS::readInputs() {
    uint8_t gain = SE_10_00V;
    int flag = fcntl(fileno(stdin), F_GETFL);
    fcntl(0, F_SETFL, flag | O_NONBLOCK);

    for (uint8_t channel = 1; channel <= 4; channel++) {
        signed short svalue = usbAIn_USB1408FS(udev, channel - 1, gain);
        std::string message = "Channel " + std::to_string(channel) + " voltage=" + std::to_string(convertToVoltage(svalue, gain)) + "V" + " pressure=" + std::to_string(convertToPsi(convertToVoltage(svalue, gain))) + "psi";
        sendData(message);
        std::cout << message << std::endl;
        usleep(1000); // Sleep for 1 milliseconds  
    }

    fcntl(fileno(stdin), F_SETFL, flag);
}

bool USB1408FS::initTCPConnection() {
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error opening socket.\n";
        return false;
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(serverPort);

    if (inet_pton(AF_INET, serverIP.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported.\n";
        return false;
    }

    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection Failed.\n";
        return false;
    }

    return true;
}

void USB1408FS::sendData(const std::string& data) {
    if (send(sockfd, data.c_str(), data.length(), 0) < 0) {
        std::cerr << "Failed to send data.\n";
    }
}

void USB1408FS::checkLibUSBError(int result, const char* context) {
    if (result < 0) {
        std::cerr << context << ": " << libusb_error_name(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

float USB1408FS::convertToVoltage(int digitalValue, uint8_t gain) const {
    return volts_1408FS_SE(digitalValue);
}

float USB1408FS::convertToPsi(float voltage) const {
    float nullVoltage = 0.50; 
    float sensitivity = 0.016; // TODO: check sensitivity value

    // Subtract the null voltage
    float activeVoltage = voltage - nullVoltage;

    // Convert active voltage to pressure in psi
    float pressurePsi = activeVoltage / sensitivity;

    return pressurePsi;
}