#include <Stepper.h>
#ifdef ARDUINO_SAMD_VARIANT_COMPLIANCE
  #define SERIAL SerialUSB
#else
  #define SERIAL Serial
#endif

const int stepsPerRevolution = 20;  // change this to fit the number of steps per revolution
// for your motor

// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

void setup() {
  // set the speed at 60 rpm:
  myStepper.setSpeed(60);
  // initialize the serial port:
  Serial.begin(9600);
}

boolean newData = false;
boolean test=false;
int previous=0;
int val=0;
char buff= ' ';

void loop() {
  
  char rc;
  // step one revolution  in one direction:
  while (Serial.available() > 0 && newData==false) {
    int val = analogRead(0);
    rc = Serial.read();
    if(rc=='f'){   
   
      Serial.println("Forward");      
      myStepper.step(val - previous);
      delay(50);
    }

    // step one revolution in the other direction:
    if(rc=='b'){
      Serial.println("True");
      Serial.println("Backward");
      myStepper.step(val + previous);
      delay(50);
    }
   previous = val;
  }
  
}

