import serial
import time
import os
import json
from datetime import datetime
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# --- CONFIGURACIÓN LOCAL ---
PUERTO = 'COM3' 
BAUDIOS = 115200
NOMBRE_ARCHIVO = 'dataset_lechugas.csv'

# --- CONFIGURACIÓN AWS IOT ---
# Reemplaza esto con tu Endpoint de la pestaña Settings en AWS IoT
ENDPOINT = "a27e0pb8csnk3r-ats.iot.us-east-2.amazonaws.com"
CLIENT_ID = "Gateway_Portatil"
TOPICO = "tesis/lechugas/datos"

# Rutas a los certificados en la carpeta 'certs'
PATH_ROOT_CA = "./certs/AmazonRootCA1.pem"
PATH_PRIVATE_KEY = "./certs/llave_privada.pem.key"
PATH_CERT = "./certs/certificado.pem.crt"

def conectar_aws():
    print("\n[~] Iniciando conexión con AWS IoT Core...")
    mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
    mqtt_client.configureEndpoint(ENDPOINT, 8883)
    mqtt_client.configureCredentials(PATH_ROOT_CA, PATH_PRIVATE_KEY, PATH_CERT)
    
    # Configuraciones de reconexión y tiempos de espera
    mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
    mqtt_client.configureOfflinePublishQueueing(-1)  # Encolado infinito en offline
    mqtt_client.configureDrainingFrequency(2)
    mqtt_client.configureConnectDisconnectTimeout(10)
    mqtt_client.configureMQTTOperationTimeout(5)

    try:
        mqtt_client.connect()
        print("[+] Conectado a AWS IoT exitosamente.")
        return mqtt_client
    except Exception as e:
        print(f"[-] ADVERTENCIA: No se pudo conectar a AWS ({e}).")
        print("    El sistema guardará datos SOLO LOCALMENTE en el CSV.")
        return None

def iniciar_captura():
    # 1. Conectar a AWS primero
    mqtt_client = conectar_aws()

    # 2. Conectar al ESP32
    try:
        ser = serial.Serial(PUERTO, BAUDIOS, timeout=2)
        print(f"[+] Conectado exitosamente al hardware en {PUERTO}")
        time.sleep(2) 
    except Exception as e:
        print(f"[-] Error fatal al conectar con el puerto {PUERTO}: {e}")
        return

    archivo_existe = os.path.isfile(NOMBRE_ARCHIVO)
    with open(NOMBRE_ARCHIVO, mode='a', newline='', encoding='utf-8') as f:
        if not archivo_existe:
            f.write("Fecha_Hora,ID_Planta,Estado_Visual,Severidad,Raw_EC,N_mgkg,P_mgkg,K_mgkg,V_450nm,B_500nm,G_550nm,Y_570nm,O_600nm,R_650nm\n")

        print("\n--- SISTEMA HÍBRIDO DE CAPTURA (LOCAL + AWS) ACTIVO ---")
        print("Escribe 'salir' en el ID para cerrar el programa.\n")

        while True:
            id_planta = input("\n1. ID de la lechuga (ej. B1_L3): ")
            if id_planta.lower() == 'salir':
                break
            if id_planta.strip() == "": continue

            print("   [1] Sano  [2] Def_N  [3] Def_P  [4] Def_K")
            estado_input = input("2. Estado visual: ")
            estados = {"1": "Sano", "2": "Def_N", "3": "Def_P", "4": "Def_K"}
            estado_visual = estados.get(estado_input, "Indefinido")

            severidad = "0"
            if estado_visual not in ["Sano", "Indefinido"]:
                while True:
                    sev_input = input("3. Severidad (1-10): ")
                    if sev_input.isdigit() and 1 <= int(sev_input) <= 10:
                        severidad = sev_input
                        break
            elif estado_visual == "Sano":
                print("   Estado Sano -> Severidad 0.")

            # Disparar lectura
            ser.write((id_planta + '\n').encode('utf-8'))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Hardware midiendo...")

            respuesta = ""
            while True:
                linea = ser.readline().decode('utf-8').strip()
                if linea.startswith(id_planta): 
                    respuesta = linea.replace(f"{id_planta},", "", 1)
                    break
                elif linea != "": pass 

            if respuesta:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # --- GUARDADO LOCAL ---
                linea_csv = f"{timestamp},{id_planta},{estado_visual},{severidad},{respuesta}\n"
                f.write(linea_csv)
                f.flush() 
                print(f"[EXITO LOCAL] CSV -> {linea_csv.strip()}")

                # --- GUARDADO EN AWS ---
                if mqtt_client:
                    try:
                        # Extraer los 10 valores separados por comas enviados por el ESP32
                        valores = respuesta.split(',')
                        payload_json = {
                            "timestamp_str": timestamp,
                            "id_planta": id_planta,
                            "etiqueta": {
                                "estado_visual": estado_visual,
                                "severidad": int(severidad)
                            },
                            "sensores": {
                                "raw_ec": float(valores[0]),
                                "nutrientes": {
                                    "N": float(valores[1]),
                                    "P": float(valores[2]),
                                    "K": float(valores[3])
                                },
                                "espectro_as7262": {
                                    "V_450nm": float(valores[4]),
                                    "B_500nm": float(valores[5]),
                                    "G_550nm": float(valores[6]),
                                    "Y_570nm": float(valores[7]),
                                    "O_600nm": float(valores[8]),
                                    "R_650nm": float(valores[9])
                                }
                            }
                        }
                        # Publicar el JSON empaquetado al tópico
                        mqtt_client.publish(TOPICO, json.dumps(payload_json), 1)
                        print(f"[EXITO CLOUD] JSON enviado a AWS IoT Core.")
                    except Exception as e:
                        print(f"[-] Fallo al enviar a la nube, pero el dato local está a salvo. Error: {e}")

            else:
                print("[!] Timeout del ESP32.")

    ser.close()
    if 'mqtt_client' in locals() and mqtt_client:
        mqtt_client.disconnect()

if __name__ == "__main__":
    iniciar_captura()