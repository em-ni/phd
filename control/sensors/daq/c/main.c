#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <ctype.h>
#include <stdint.h>

#include "pmd.h"
#include "usb-1408FS.h"


int main (int argc, char **argv)
{
  int flag;
  signed short svalue;
  uint8_t channel, gain;
  libusb_device_handle *udev = NULL;
  int ret;

  ret = libusb_init(NULL);
  if (ret < 0) {
    perror("libusb_init: Failed to initialize libusb");
    exit(1);
  }

  if ((udev = usb_device_find_USB_MCC(USB1408FS_PID, NULL))) {
    printf("USB-1408FS Device is found!\n");
  } else {
    printf("No device found.\n");
    exit(0);
  }

  init_USB1408FS(udev);
  
  /* config mask 0x01 means all inputs */
  usbDConfigPort_USB1408FS(udev, DIO_PORTA, DIO_DIR_OUT);
  usbDConfigPort_USB1408FS(udev, DIO_PORTB, DIO_DIR_IN);
  usbDOut_USB1408FS(udev, DIO_PORTA, 0);
  usbDOut_USB1408FS(udev, DIO_PORTA, 0);

  printf("Reading inputs\n");

  while(1) {
    gain =  SE_10_00V;
    flag = fcntl(fileno(stdin), F_GETFL);
    fcntl(0, F_SETFL, flag | O_NONBLOCK);

    // Read channel 1
    channel = 1;
    svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    sleep(1);  // Delay for stability

    // Read channel 2
    channel = 2;
    svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    sleep(1);  // Delay for stability

    // Read channel 3
    channel = 3;
    svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    sleep(1);  // Delay for stability

    // Read channel 4
    channel = 4;
    svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    sleep(1);  // Delay for stability

    // // Read channel 5
    // channel = 5;
    // svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    // printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    // sleep(1);  // Delay for stability

    // // Read channel 6
    // channel = 6;
    // svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    // printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    // sleep(1);  // Delay for stability

    // // Read channel 7
    // channel = 7;
    // svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    // printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    // sleep(1);  // Delay for stability

    // // Read channel 8
    // channel = 8;
    // svalue = usbAIn_USB1408FS(udev, channel-1, gain);
    // printf("Channel %d: value=%.2fV\n",channel, volts_1408FS_SE(svalue));
    // sleep(1);  // Delay for stability
    

    fcntl(fileno(stdin), F_SETFL, flag);
  }

  libusb_close(udev);
  libusb_exit(NULL);
  return 0;
}

