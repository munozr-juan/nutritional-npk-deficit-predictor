#include <Arduino.h>
#include <ModbusMaster.h>
#include <Wire.h>
#include <Adafruit_AS726x.h>

// --- HARDWARE CONFIGURATION ---
#define MAX485_RE_DE 4
#define RX2_PIN 16
#define TX2_PIN 17
#define SAMPLES_PER_BURST 10 

ModbusMaster node;
Adafruit_AS726x ams;

// --- FUNCTION PROTOTYPES ---
void preTransmission();
void postTransmission();
void executeBurst(String id);
bool recoverOpticalSensor(); 

// --- TRANSCEIVER DIRECTION CONTROL ---
void preTransmission() { digitalWrite(MAX485_RE_DE, HIGH); }
void postTransmission() { delay(5); digitalWrite(MAX485_RE_DE, LOW); }

// I2C Bus recovery routine to resolve sensor lockups
bool recoverOpticalSensor() {
  Wire.end(); 
  delay(10);
  Wire.begin(21, 22); 
  Wire.setClock(100000);
  
  if(ams.begin()) {
    ams.setGain(GAIN_64X); 
    ams.setIntegrationTime(50);
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  
  pinMode(MAX485_RE_DE, OUTPUT);
  digitalWrite(MAX485_RE_DE, LOW);
  Serial2.begin(9600, SERIAL_8N1, RX2_PIN, TX2_PIN);
  
  node.begin(1, Serial2);
  node.preTransmission(preTransmission);
  node.postTransmission(postTransmission);

  if(!recoverOpticalSensor()){
    Serial.println("CRITICAL_ERROR_AS7262_INIT_FAILED");
    while(1);
  }

  delay(200);
  while(Serial.available()){ Serial.read(); } // Clear serial buffer baseline

  Serial.println("\n--- EMBEDDED SYSTEM READY FOR INGESTION ---");
  Serial.println("Waiting for ingestion pipeline token...");
}

void loop() {
  if (Serial.available() > 0) {
    String plantId = Serial.readStringUntil('\n');
    String cleanId = "";
    
    // Sanitize string from USB line noise or phantom characters
    for(int i=0; i<plantId.length(); i++){
      char c = plantId.charAt(i);
      if(isAlphaNumeric(c) || c == '_' || c == '-') {
        cleanId += c;
      }
    }

    if (cleanId.length() > 2) {
      executeBurst(cleanId);
    }
  }
}

void executeBurst(String id) {
  long finalRawEC = -1, finalN = -1, finalP = -1, finalK = -1;
  float sumV = 0, sumB = 0, sumG = 0, sumY = 0, sumO = 0, sumR = 0;

  ams.drvOn(); // Turn on built-in LED indicator for stabilization
  unsigned long startWarming = millis();

  // Asynchronous Modbus Poll during LED thermal stabilization window
  bool modbusOk = false;
  while (!modbusOk && (millis() - startWarming < 3000)) {
    bool ecOk = false, npkOk = false;

    if (node.readHoldingRegisters(0x0007, 1) == node.ku8MBSuccess) {
      finalRawEC = node.getResponseBuffer(0);
      ecOk = true;
    }
    delay(30);
    if (node.readHoldingRegisters(0x001E, 3) == node.ku8MBSuccess) {
      finalN = node.getResponseBuffer(0);
      finalP = node.getResponseBuffer(1);
      finalK = node.getResponseBuffer(2);
      npkOk = true;
    }

    if (ecOk && npkOk) modbusOk = true; 
    else delay(200); 
  }

  while (millis() - startWarming < 3000) delay(10); // Maintain sync

  // --- SPECTRAL ACQUISITION BURST ---
  int validOpticalReadings = 0;

  for (int i = 0; i < SAMPLES_PER_BURST; i++) {
    ams.startMeasurement(); 
    delay(150); // Hard integration delay requirement
    
    unsigned long waitSensor = millis();
    bool isReady = false;
    
    while(!isReady && (millis() - waitSensor < 800)) {
        if(ams.dataReady()) isReady = true;
        else delay(10);
    }
    
    if (isReady) {
        float v = ams.readViolet();
        // Spurious noise logic filter
        if (v > 0.1 && v < 15000.0) {
            sumV += v;
            sumB += ams.readBlue();
            sumG += ams.readGreen();
            sumY += ams.readYellow();
            sumO += ams.readOrange();
            sumR += ams.readRed();
            validOpticalReadings++;
        } else {
            isReady = false; 
        }
    } 
    
    if (!isReady) {
        recoverOpticalSensor(); // Hot-plug I2C restoration on bus lockup
        delay(50); 
    }
  }

  ams.drvOff();

  float avgV = 0, avgB = 0, avgG = 0, avgY = 0, avgO = 0, avgR = 0;
  if (validOpticalReadings > 0) {
      avgV = sumV / validOpticalReadings;
      avgB = sumB / validOpticalReadings;
      avgG = sumG / validOpticalReadings;
      avgY = sumY / validOpticalReadings;
      avgO = sumO / validOpticalReadings;
      avgR = sumR / validOpticalReadings;
  }

  // Comma-separated output frame packet pipeline stream
  Serial.print(id); Serial.print(",");
  Serial.print(finalRawEC); Serial.print(",");
  Serial.print(finalN); Serial.print(",");
  Serial.print(finalP); Serial.print(",");
  Serial.print(finalK); Serial.print(",");
  Serial.print(avgV, 2); Serial.print(",");
  Serial.print(avgB, 2); Serial.print(",");
  Serial.print(avgG, 2); Serial.print(",");
  Serial.print(avgY, 2); Serial.print(",");
  Serial.print(avgO, 2); Serial.print(",");
  Serial.println(avgR, 2);
}