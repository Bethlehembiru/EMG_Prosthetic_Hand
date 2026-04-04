#include <Servo.h>

// -----------------------------
// CONFIGURATION
// -----------------------------
const int BAUD_RATE = 115200;

// Servo objects
Servo thumbServo;
Servo indexServo;
Servo multiServo;

// Servo pins
const int THUMB_PIN = 9;
const int INDEX_PIN = 10;
const int MULTI_PIN = 11;

// Gesture mapping (example positions)
struct GesturePos {
  int thumb;
  int index;
  int multi;
};

// Define positions for each gesture
GesturePos gestures[4] = {
  {90, 90, 90},    // 0: Rest
  {0, 0, 0},       // 1: Close
  {180, 180, 180}, // 2: Open
  {45, 0, 90}      // 3: Point (example)
};

void setup() {
  Serial.begin(BAUD_RATE);

  // Attach servos
  thumbServo.attach(THUMB_PIN);
  indexServo.attach(INDEX_PIN);
  multiServo.attach(MULTI_PIN);

  // Start at rest
  thumbServo.write(gestures[0].thumb);
  indexServo.write(gestures[0].index);
  multiServo.write(gestures[0].multi);

  Serial.println("Actuation Ready");
}

void loop() {
  if (Serial.available()) {
    int gesture = Serial.parseInt();  // Read gesture ID from controller

    if (gesture >= 0 && gesture <= 3) {
      thumbServo.write(gestures[gesture].thumb);
      indexServo.write(gestures[gesture].index);
      multiServo.write(gestures[gesture].multi);

      Serial.print("Gesture: "); Serial.println(gesture);
    } else {
      Serial.println("Unknown gesture");
    }
  }
}














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