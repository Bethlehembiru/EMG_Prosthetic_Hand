#include <Servo.h>

Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;

String incoming = "";


// =====================================================
// SETUP
// =====================================================
void setup()
{
    Serial.begin(9600);

    // Attach servos
    thumbServo.attach(3);
    indexServo.attach(5);
    middleServo.attach(6);
    ringServo.attach(9);
    pinkyServo.attach(10);

    // Initial neutral position
    thumbServo.write(90);
    indexServo.write(90);
    middleServo.write(90);
    ringServo.write(90);
    pinkyServo.write(90);

    Serial.println("5-Servo Hand Ready");
}


// =====================================================
// MAIN LOOP
// =====================================================
void loop()
{
    while (Serial.available())
    {
        char c = Serial.read();

        if (c == '\n')
        {
            processCommand(incoming);
            incoming = "";
        }
        else
        {
            incoming += c;
        }
    }
}


// =====================================================
// PROCESS SERIAL COMMAND
// Format:
// thumb,index,middle,ring,pinky
// Example:
// 0,0,0,0,0
// =====================================================
void processCommand(String cmd)
{
    int commas[4];

    commas[0] = cmd.indexOf(',');
    commas[1] = cmd.indexOf(',', commas[0] + 1);
    commas[2] = cmd.indexOf(',', commas[1] + 1);
    commas[3] = cmd.indexOf(',', commas[2] + 1);

    // Validation
    if (
        commas[0] == -1 ||
        commas[1] == -1 ||
        commas[2] == -1 ||
        commas[3] == -1
    )
    {
        return;
    }

    int thumb =
        cmd.substring(
            0,
            commas[0]
        ).toInt();

    int index =
        cmd.substring(
            commas[0] + 1,
            commas[1]
        ).toInt();

    int middle =
        cmd.substring(
            commas[1] + 1,
            commas[2]
        ).toInt();

    int ring =
        cmd.substring(
            commas[2] + 1,
            commas[3]
        ).toInt();

    int pinky =
        cmd.substring(
            commas[3] + 1
        ).toInt();


    // =================================================
    // MOVE SERVOS
    // =================================================
    thumbServo.write(thumb);
    indexServo.write(index);
    middleServo.write(middle);
    ringServo.write(ring);
    pinkyServo.write(pinky);


    // =================================================
    // DEBUG OUTPUT
    // =================================================
    Serial.print("Thumb: ");
    Serial.print(thumb);

    Serial.print(" | Index: ");
    Serial.print(index);

    Serial.print(" | Middle: ");
    Serial.print(middle);

    Serial.print(" | Ring: ");
    Serial.print(ring);

    Serial.print(" | Pinky: ");
    Serial.println(pinky);
}