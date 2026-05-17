const int emgPin = A3;

// 1 kHz target → 1000 microseconds
const unsigned long sampleInterval = 1000;

unsigned long lastTime = 0;

void setup() {
  Serial.begin(9600);  // higher baud = smoother streaming
}

void loop() {
  unsigned long now = micros();

  if (now - lastTime >= sampleInterval) {
    lastTime = now;

    int emgValue = analogRead(emgPin);

    Serial.println(emgValue);
  }
}