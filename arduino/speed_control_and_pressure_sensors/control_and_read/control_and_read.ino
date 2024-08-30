
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
bool printRaw = false;
bool printVoltage = false;
bool printPressure = false;

unsigned long lastSensorReadTime = 0;
const long sensorReadInterval = 2000;  // Sensor read interval in milliseconds


float adcToPressureMPa(int adcValue, int sensorNumber) {
  // Sensor data  
  float referenceVoltage = 5.0; // Assuming 5V reference voltage
  float nullVoltage = 0.50;     // Sensor's null voltage
  float sensitivity = 0.016;   // Sensor's sensitivity in V/psi
  float psiToMPa = 0.00689476;  // Conversion factor from psi to MPa
  
  // Convert ADC value to voltage
  float voltage = (adcValue / 1023.0) * referenceVoltage;
  
  // Subtract the null voltage
  float activeVoltage = voltage - nullVoltage;

  if (printVoltage == true){
    Serial.print("Voltage: ");
    Serial.println(activeVoltage);
  }
  
  // Convert active voltage to pressure in psi
  float pressurePsi = activeVoltage / sensitivity;
  
  // Convert pressure from psi to MPa
  float pressureMPa = pressurePsi * psiToMPa;

  // Print the pressure value in MPa to the serial monitor
  if (printPressure == true){
    Serial.print("Pressure ");
    Serial.print(sensorNumber);
    Serial.print(" :");
    Serial.print(pressureMPa, 3); // Print with 3 decimal places
    Serial.println(" MPa");
  }
  
  return pressureMPa;
}



void setup() {
   Serial.begin(9600);
}

void loop() {   

  unsigned long currentMillis = millis();

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
  int sensorReading = analogRead(A0);
  // map it to a range from 0 to 100:
  int motorSpeed = map(sensorReading, 0, 1023, 0, 100);
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
  if (currentMillis - lastSensorReadTime >= sensorReadInterval) {
    lastSensorReadTime = currentMillis;
    // Read the value from the sensors
    int valueA1 = analogRead(A1);
    int valueA2 = analogRead(A2);
    int valueA3 = analogRead(A3);
    int valueA4 = analogRead(A4);
  
    if (printRaw == true){
      Serial.println("Raw: ");
      Serial.println(valueA1);
      Serial.println(valueA2);
      Serial.println(valueA3);
      Serial.println(valueA4);
    }
  
    float pressureA1 = adcToPressureMPa(valueA1, 1);
    float pressureA2 = adcToPressureMPa(valueA2, 2);
    float pressureA3 = adcToPressureMPa(valueA3, 3);
    float pressureA4 = adcToPressureMPa(valueA4, 4);
  }
  

}
