# python 3.11
import random
import time
import pandas as pd
import os
import glob
import json
from paho.mqtt import client as mqtt_client


# Set the number of gateways to 2, 20, or 50
number_of_gateways = 2

if number_of_gateways == 2:
    dataset_folder = './roma-2-gw/output/merged_tx_rx'
    #gateways_list_file = './roma-2-gw/output/2gatewaysMAC.txt'

elif number_of_gateways == 20:
    dataset_folder = './roma-20-gw/output/merged_tx_rx'
    #gateways_list_file = './roma-20-gw/output/20gatewaysMAC.txt'

elif number_of_gateways == 50:
    dataset_folder = './roma-50-gw/output/merged_tx_rx'
    #gateways_list_file = './roma-50-gw/output/50gatewaysMAC.txt'
else:
    raise Exception("Invalid number of gateways")


snapshots = glob.glob(f'{dataset_folder}/*.csv')
print(f"Found {len(snapshots)} CSV files in {dataset_folder}")

# Setup MQTT Broker connection details
broker = 'localhost'
port = 1883
topic = "e2l/packets"
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'
username = 'user'
password = 'password'

# Connect to the MQTT Broker
def connect_mqtt():
    def on_connect(client, userdata, flags, reasonCode, properties=None):
        if reasonCode == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {reasonCode}")

    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2, protocol=mqtt_client.MQTTv5)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

# Publish message to MQTT Broker
def publish(client, message):
    time.sleep(0.01)
    result = client.publish(topic, message)
    status = result.rc
    if status == 0:
        print(f"Sent {message} to topic {topic}")
    else:
        print(f"Failed to send message to topic {topic}: {status}")

# Run the MQTT Client
def run():
    client = connect_mqtt()
    print(f"Connecting to {broker}")
    client.loop_start()
    for snapshot in snapshots:
        print(f"Processing file: {snapshot}")
        df = pd.read_csv(snapshot)
        for index, row in df.iterrows():
            if row['receptions'] == "[]":
                continue
            message = row.to_json()
            row['receptions'] = eval(row['receptions'])
            RX_GW = []

            for reception in row['receptions']:
                RX_GW.append(reception[7])
                """
                message = {
                    "gw_mac_address": reception[7]
                }
                """
            message = {
                    "x_coordinate": row['x'],
                    "y_coordinate": row['y'],
                    "dev_address": row['dev_addr'],
                    #"label": reception[0],
                    #"timestamp": reception[1],
                    #"node_id": row['NODE_ID'],
                    "spreading_factor": row['spreading_factor'],
                    #"frame_counter": reception[4],
                    #"frequency": reception[5],
                    #"rssi": reception[6],
                    "Receiving gateways" : RX_GW
                }
            
            message = json.dumps(message)
            publish(client, message)

    client.loop_stop()

if __name__ == '__main__':
    run()