
/*
 Stepper Motor Control - speed control

 This program drives a unipolar or bipolar stepper motor.
 The motor is attached to digital pins 8 - 11 of the Arduino.
 A potentiometer is connected to analog input 0.

 The motor will rotate in a clockwise direction. The higher the potentiometer value,
 the faster the motor speed. Because setSpeed() sets the delay between steps,
 you may notice the motor is less responsive to changes in the sensor vaflue at
 low speeds.

 Created 30 Nov. 2009
 Modified 28 Oct 2010
 by Tom Igoe

 */

#include <Stepper.h>

const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution


// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

int stepCount = 0;  // number of steps the motor has taken

// character to read Serial
char rc;

// Print variables
bool printForward = false;
bool printBackward = false;
bool printStopped = false;



void setup() {
   Serial.begin(9600);
}

void loop() {   

  // If Serial is triggered get Serial signal in temporary variable
  char temp;
  if (Serial.available() > 0) {
    Serial.println("Reading Serial");
    temp = Serial.read();
    Serial.println(temp);

    // Assign Serial signal only if it is a known command
    if (temp == 'f' || temp == 'b' || temp == 's'){
      rc = temp;
      Serial.println("got signal");
      Serial.println(rc);
    }
  }

  // Print status
  if(rc=='f' && printForward == false){
    Serial.println("Forward");
    printForward = true;
    printBackward = false;
    printStopped = false;
  }
  if(rc=='b' && printBackward == false){
    Serial.println("Backward");
    printForward = false;
    printBackward = true;
    printStopped = false;
  }
  if(rc=='s' && printStopped == false){
    Serial.println("Stopped");
    printForward = false;
    printBackward = false;
    printStopped = true;
  }
  
  // Move motor with potentiometer
//  int sensorReading = analogRead(A0);
//  // map it to a range from 0 to 100:
//  int motorSpeed = map(sensorReading, 0, 1023, 0, 100);
  int motorSpeed = 80;
  // set the motor speed:
  if (motorSpeed > 0) {
    myStepper.setSpeed(motorSpeed);

    if (rc == 'f'){
      // step 1/100 of a revolution:
      myStepper.step(stepsPerRevolution / 100);
    }
    if (rc == 'b'){
        // step 1/100 of a revolution:
      myStepper.step(-stepsPerRevolution / 100);
    }

    if (rc == 's'){
      // zero velocity:
      myStepper.step(0);
    }
  
  }

}
