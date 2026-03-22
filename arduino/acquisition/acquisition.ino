#include <MyoWare.h>

// MyoWare class object
MyoWare myoware;

// the setup routine runs once when you press reset:
void setup() 
{
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);

  // output conversion parameters - modify these values to match your setup
  myoware.setConvertOutput(true);     // Set to true to convert ADC output to the amplitude of
                                      // of the muscle activity as it appears at the electrodes
                                      // in millivolts
  myoware.setADCResolution(12.);      // ADC bits (shield default = 12-bit)
  myoware.setADCVoltage(3.3);         // ADC reference voltage (shield default = 3.3V)
  myoware.setGainPotentiometer(50.);  // Gain potentiometer resistance in kOhms.
                                      // adjust the potentiometer setting such that the
                                      // max muscle reading is below 3.3V then update this
                                      // parameter to the measured value of the potentiometer
  myoware.setENVPin(A0);              // Arduino pin connected to ENV
  myoware.setRAWPin(A1);              // Arduino pin connected to RAW
  myoware.setREFPin(A2);              // Arduino pin connected to REF
  myoware.setRECTPin(A3);             // Arduino pin connected to RECT
}

// the loop routine runs over and over again forever:
void loop() 
{
  // read the sensor's analog output pins  
  const double envMillivolts = myoware.readSensorOutput(MyoWare::ENVELOPE);
  const double rawMillivolts = myoware.readSensorOutput(MyoWare::RAW);
  const double rectMillivolts = myoware.readSensorOutput(MyoWare::RECTIFIED);
  
  // print output in millivolts:
  Serial.print(envMillivolts);
  Serial.print(",");
  Serial.print(rawMillivolts);
  Serial.print(",");
  Serial.println(rectMillivolts);
