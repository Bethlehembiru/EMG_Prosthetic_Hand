#include <MyoWare.h>
MyoWare myoware;

// CONFIG
const int BAUD_RATE = 115200;
const int SAMPLE_RATE = 1000;
const long SAMPLE_INTERVAL = 1000000 / SAMPLE_RATE;

const int EMG_PIN = A1;

unsigned long previousMicros = 0;
bool isStreaming = false;

void setup(){
  Serial.begin(BAUD_RATE);

  // MyoWare setup
  myoware.setConvertOutput(false);
  myoware.setRAWPin(EMG_PIN);

  Serial.println("READY");
}

void loop(){
  handleSerial();

  if (!isStreaming) return;

  unsigned long currentMicros = micros();

  if (currentMicros - previousMicros >= SAMPLE_INTERVAL)
  {
    previousMicros += SAMPLE_INTERVAL;

    int rawValue = analogRead(EMG_PIN);
    Serial.println(rawValue);
  }
}

// COMMAND HANDLER
void handleSerial(){
  if (Serial.available())
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "START")
    {
      isStreaming = true;
      previousMicros = micros();
      Serial.println("STREAMING");
    }
    else if (cmd == "STOP")
    {
      isStreaming = false;
      Serial.println("STOPPED");
    }
  }
}
