#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(115200);
  myServo.attach(9);
}

void loop() {
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();
    myServo.write(angle);
  }
}