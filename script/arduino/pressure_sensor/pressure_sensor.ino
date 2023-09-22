void setup() {
  // Start serial communication at 9600 baud rate
  Serial.begin(9600);
}

void loop() {
  // Read the value from the sensor connected to A0
  int valueA0 = analogRead(A0);
  
  // Read the value from the sensor connected to A1
  int valueA1 = analogRead(A1);
  
  // Print the values to the serial monitor
  Serial.print("Value from A0: ");
  Serial.println(valueA0);
  Serial.print("Value from A1: ");
  Serial.println(valueA1);
  
  // Delay for a bit before reading again
  delay(1000); // 1 second delay
}
