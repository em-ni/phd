void setup() {
  // Start serial communication at 9600 baud rate
  Serial.begin(9600);
}

float adcToPressureMPa(int adcValue) {
  float referenceVoltage = 5.0; // Assuming 5V reference voltage
  float nullVoltage = 0.50;     // Sensor's null voltage
  float sensitivity = 0.016;   // Sensor's sensitivity in V/psi
  float psiToMPa = 0.00689476;  // Conversion factor from psi to MPa
  
  // Convert ADC value to voltage
  float voltage = (adcValue / 1023.0) * referenceVoltage;
  
  // Subtract the null voltage
  float activeVoltage = voltage - nullVoltage;
  
  // Convert active voltage to pressure in psi
  float pressurePsi = activeVoltage / sensitivity;
  
  // Convert pressure from psi to MPa
  float pressureMPa = pressurePsi * psiToMPa;
  
  return pressureMPa;
}

void loop() {
  // Read the value from the sensor connected to A0
  int valueA0 = analogRead(A0);
  float pressureA0 = adcToPressureMPa(valueA0);
  
  // Print the pressure value in MPa to the serial monitor
  Serial.print("Pressure from A0: ");
  Serial.print(pressureA0, 3); // Print with 3 decimal places
  Serial.println(" MPa");
  
  // Delay for a bit before reading again
  delay(1000); // 1 second delay
}
