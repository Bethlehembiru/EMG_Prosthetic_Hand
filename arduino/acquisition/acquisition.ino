 void setup() {
  Serial.begin(9600);
}

void loop() {
  int emgValue = analogRead(A0);
  Serial.println(emgValue);
  delay(10);