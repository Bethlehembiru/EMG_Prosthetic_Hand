#include <MyoWare.h>

MyoWare myoware;

// CONFIG
const int BAUD_RATE = 115200;
const int SAMPLE_RATE = 1000;
const long SAMPLE_INTERVAL = 1000000 / SAMPLE_RATE;

unsigned long previousMicros = 0;
bool isStreaming = false;

void setup()
{
  Serial.begin(BAUD_RATE);
  while (!Serial);

  // MyoWare setup
  myoware.setConvertOutput(false);   // IMPORTANT → raw ADC
  myoware.setRAWPin(A1);

  Serial.println("READY");
}

void loop()
{
  handleSerial();

  if (!isStreaming) return;

  unsigned long currentMicros = micros();

  if (currentMicros - previousMicros >= SAMPLE_INTERVAL)
  {
    previousMicros += SAMPLE_INTERVAL;

    int rawValue = analogRead(A1);   // direct RAW read

    Serial.println(rawValue);
  }
}

// COMMAND HANDLER
void handleSerial()
{
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
