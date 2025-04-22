CC=gcc
CFLAGS+= -g -Wall -fPIC -O -I/usr/local/include/libusb-1.0 -L/usr/local/lib -lusb-1.0
LDFLAGS=-lusb-1.0 -lhidapi-libusb
SOURCES=main.c usb-1408FS.c pmd.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=read_sensors

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

.PHONY: all clean
