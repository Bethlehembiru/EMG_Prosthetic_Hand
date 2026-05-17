#include <Servo.h>

// =====================================================
// EMG INPUT
// =====================================================
const int emgPin = A3;
const unsigned long sampleInterval = 1000;
unsigned long lastTime = 0;


// =====================================================
// SERVOS
// =====================================================
Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;


// =====================================================
// SERIAL BUFFER
// =====================================================
String incoming = "";


// =====================================================
// SETUP
// =====================================================
void setup() {

    Serial.begin(9600);

    // attach servos
    thumbServo.attach(3);
    indexServo.attach(5);
    middleServo.attach(6);
    ringServo.attach(9);
    pinkyServo.attach(10);

    // neutral
    thumbServo.write(90);
    indexServo.write(90);
    middleServo.write(90);
    ringServo.write(90);
    pinkyServo.write(90);

    Serial.println("READY");
}


// =====================================================
// MAIN LOOP
// =====================================================
void loop() {

    // =============================================
    // 1. SEND EMG SAMPLE (NON-BLOCKING)
    // =============================================
    unsigned long now = micros();

    if (now - lastTime >= sampleInterval) {
        lastTime = now;

        int emgValue = analogRead(emgPin);

        Serial.print("E:");
        Serial.println(emgValue);
    }


    // =============================================
    // 2. READ SERIAL COMMANDS
    // =============================================
    while (Serial.available()) {

        char c = Serial.read();

        if (c == '\n') {

            processCommand(incoming);
            incoming = "";

        } else {
            incoming += c;
        }
    }
}


// =====================================================
// PROCESS CONTROL COMMAND
// Format:
// C:thumb,index,middle,ring,pinky
// =====================================================
void processCommand(String cmd) {

    if (!cmd.startsWith("C:")) return;

    cmd = cmd.substring(2); // remove "C:"

    int commas[4];

    commas[0] = cmd.indexOf(',');
    commas[1] = cmd.indexOf(',', commas[0] + 1);
    commas[2] = cmd.indexOf(',', commas[1] + 1);
    commas[3] = cmd.indexOf(',', commas[2] + 1);

    if (commas[3] == -1) return;

    int thumb = cmd.substring(0, commas[0]).toInt();
    int index = cmd.substring(commas[0] + 1, commas[1]).toInt();
    int middle = cmd.substring(commas[1] + 1, commas[2]).toInt();
    int ring = cmd.substring(commas[2] + 1, commas[3]).toInt();
    int pinky = cmd.substring(commas[3] + 1).toInt();

    thumbServo.write(thumb);
    indexServo.write(index);
    middleServo.write(middle);
    ringServo.write(ring);
    pinkyServo.write(pinky);
}