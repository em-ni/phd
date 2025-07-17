void setup() {
  // Start serial communication at 9600 baud rate
  Serial.begin(9600);
}

float adcToPressureMPa(int adcValue, int sensorNumber) {
  bool printVoltage = false;
  bool printPressure = true;
  
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

void loop() {
  bool printRaw = false;
  
  // Read the value from the sensor connected to A0
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
  

  Serial.println(" ");
  
  // Delay for a bit before reading again
  delay(1000); // 1 second delay
}
