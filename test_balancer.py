from calendar import c
import random
import json
from turtle import distance, update
from paho.mqtt import client as mqtt_client
import pandas as pd
import math
import numpy as np
from collections import defaultdict
import threading
import pickle
import os

from requests import get

from score_logger import ScoreLogger
from statistics import mean 

# Setup MQTT Broker connection details
broker = 'localhost'
port = 1883
topic = "e2l/packets"
client_id = f'subscribe-{random.randint(0, 100)}'
username = 'user'
password = 'password'

# Define RL parameters
alpha = 0.1  
gamma = 0.95  
epsilon = 0.1  
# Set the number and resources of gateways to 2, 20, or 50
number_of_gateways = 50
GWResource = 300
gateway_powers = {i: GWResource for i in range(number_of_gateways)}

# variables for the score logger
num_messages = 0

score_log = ScoreLogger("TOTAL_COST", 1000)

if number_of_gateways == 2:
    dataset_folder = './roma-2-gw/output/merged_tx_rx'
    gateways_list_file = './roma-2-gw/2gatewaysMAC.txt'
    gateways_positions_file = './roma-2-gw/gw-conf/gw-roma-2_ns3.csv'

elif number_of_gateways == 20:
    dataset_folder = './roma-20-gw/output/merged_tx_rx'
    gateways_list_file = './roma-20-gw/20gatewaysMAC.txt'
    gateways_positions_file = './roma-20-gw/gw_info/gw-roma-20_ns3.csv'

elif number_of_gateways == 50:
    dataset_folder = './roma-50-gw/output/merged_tx_rx'
    gateways_list_file = './roma-50-gw/50gatewaysMAC.txt'
    gateways_positions_file = './roma-50-gw/gw-conf/gw-roma-50.csv'
else:
    raise Exception("Invalid number of gateways")

def save_variable(table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(table, f)

def load_variable(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def haversine(lat1, lon1, lat2, lon2):
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

# Load the list of gateways
with open(gateways_list_file, 'r') as f:
    gateways = [line.strip() for line in f]

"""
print("The unsorted list of gateways MAC IDs: ", gateways)
gateways.sort()
print("The sorted list of gateways MAC IDs: ", gateways)
"""

gateways_positions = pd.read_csv(gateways_positions_file)
#print(gateways_positions)

class DefaultDict(defaultdict):
        def __missing__(self, key):
            self[key] = defaultdict(float)
            return self[key]

if os.path.exists('q_table.pkl'):
    q_table = load_variable('q_table.pkl')
else:
    q_table = DefaultDict(defaultdict)

active_tasks = {}

def reset_payload_cost(gateway_idx, payload_cost, mex_idx):
    global gateway_powers
    
    gateway_powers[gateway_idx] += payload_cost
    if gateway_powers[gateway_idx] > GWResource:
        gateway_powers[gateway_idx] = GWResource  
    print(f"Task {mex_idx} completed for Gateway {gateway_idx}. Reset power by {payload_cost}. Current power: {gateway_powers[gateway_idx]}")

    # Safely remove the task from active_tasks
    if mex_idx in active_tasks:
        del active_tasks[mex_idx]
    else:
        print(f"Warning: message {mex_idx} not found in active tasks.")


def start_task_timer(mex_idx, gateway_idx, payload_cost, time):
    timer = threading.Timer(time, reset_payload_cost, args=(gateway_idx, payload_cost, mex_idx))
    timer.start()
    active_tasks[mex_idx] = {"gateway": gateway_idx, "payload_cost": payload_cost, "timer": timer}

# Connect to the MQTT Broker
def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, reasonCode, properties=None):
        if reasonCode == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", reasonCode)

    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2, protocol=mqtt_client.MQTTv5)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def get_state(near_gw, operation_cost):
        
    resource_tuple = tuple(int(gateway_powers[i]/100) for i in range(number_of_gateways))
    return (resource_tuple, near_gw, operation_cost)
    
def get_possible_actions(operation_cost):
    
    return [i for i in range(number_of_gateways) 
            if gateway_powers[i] >= operation_cost]

def get_action(state, possible_actions):
    
    if not possible_actions:
        return -1
    else:
        if np.random.random() < epsilon:
            
            return np.random.choice(possible_actions)
        else:
            
            return get_best_action(state, possible_actions)

def get_best_near_gw(near_gw):

    gw = near_gw[0]
    for i in near_gw:
        j = i + 1
        for j in near_gw:
            if i != j:
                if gateway_powers[i] > gateway_powers[j]:
                    print ("gw:", i, "power:", gateway_powers[i], "gw:", j, "power:", gateway_powers[j])
                    gw = i
    if gateway_powers[gw] <= 0:
        return -1
    else:
        return gw
    
def choose_random_gw(payload_cost):
    random_gw = np.random.choice(range(number_of_gateways))
    if gateway_powers[random_gw] < payload_cost:
        return -1
    else:
        return random_gw
    """
    if not possible_actions:
        return -1
    else:
        return np.random.choice(possible_actions)
    """


def get_best_action(state, possible_actions):
   
    q_values = [q_table[state][action] for action in possible_actions]
    max_q = max(q_values)
    best_actions = [action for action, q_value in zip(possible_actions, q_values) 
                    if q_value == max_q]
    return np.random.choice(best_actions)

def calculate_reward(gateway_idx, near_gw, operation_cost):

    remaining_power = gateway_powers[gateway_idx] - operation_cost
    #print (gateway_idx)

    if(gateway_idx in near_gw):
        return  1 #- (remaining_power / GWResource)
    else:
        return -1
    
def update(state, action, reward, next_state, next_possible_actions):
    
    if next_possible_actions:
        next_q = max(q_table[next_state][next_action] 
                    for next_action in next_possible_actions)
    else:
        next_q = 0
        
    current_q = q_table[state][action]
    q_table[state][action] = current_q + alpha * (reward + gamma * next_q - current_q)

def update_gateway_power(gateway_idx, operation_cost):
    
    gateway_powers[gateway_idx] -= operation_cost
    
def reset_gateway_powers():
    
    gateway_powers = {i: GWResource for i in range(number_of_gateways)}

def get_distances(x_ed, y_ed):
    distances = []
    for i in range(number_of_gateways):
        if(number_of_gateways == 50 or number_of_gateways == 2):
            distances.append(int(haversine(x_ed, y_ed, gateways_positions['lat'][i], gateways_positions['lon'][i])/1000))
        else:
            distances.append(int(haversine(x_ed, y_ed, gateways_positions['X'][i], gateways_positions['Y'][i])/1000))
    return distances

def intelligentJointAlgorithm(msg):
    message = json.loads(msg)
    payload_cost = random.choice([5, 10])
    sf = message['spreading_factor']
    global num_messages
    num_messages += 1
    """
    if sf == 7:
        payload_cost += 0.00184
    elif sf == 8:
        payload_cost += 0.00328
    elif sf == 9:
        payload_cost += 0.00584
    elif sf == 10:
        payload_cost += 0.0106
    elif sf == 11:
        payload_cost += 0.0193
    elif sf == 12:
        payload_cost += 0.0360
    """
    distances = tuple(get_distances(message['x_coordinate'], message['y_coordinate']))
    #print(f"Distances: {distances}")

    rx_gw = message['Receiving gateways']
    #print(f"Received gateways: {rx_gw}")

    near_gw = []
    for i in range(number_of_gateways):
        if gateways[i] in rx_gw:
            near_gw.append(i)
    near_gw = tuple(near_gw)
    print (near_gw)

    current_state = get_state(near_gw, payload_cost)
    #print(f"Current state: {current_state}")
    possible_actions = get_possible_actions(payload_cost)

    selected_gateway = get_action(current_state, possible_actions)

    #selected_gateway = get_best_near_gw(near_gw)

    #selected_gateway = choose_random_gw(payload_cost)

    if selected_gateway == -1:
        reward = -20
        TOTAL_COST = 5000
    else:
        update_gateway_power(selected_gateway, payload_cost)
        reward = calculate_reward(selected_gateway, near_gw, payload_cost)
        if(selected_gateway in near_gw):
            TOTAL_COST = payload_cost
        else:
            TOTAL_COST = payload_cost + distances[selected_gateway]
        
    
    print(f"Total cost: {TOTAL_COST}, selected gateway: {selected_gateway}, reward: {reward}")

    score_log.add_score(int(TOTAL_COST), num_messages)

    next_state = get_state(near_gw, payload_cost)
    #print(f"Next state: {next_state}")
    next_possible_actions = get_possible_actions(payload_cost)
    update(current_state, selected_gateway, reward, next_state, next_possible_actions)
    save_variable(q_table, 'q_table.pkl')

    if number_of_gateways == 50:
        life_time = (sf + payload_cost)
    elif number_of_gateways == 20:
        life_time = (sf + payload_cost) * 0.5
    elif number_of_gateways == 2:
        life_time = (sf + payload_cost) * 0.04

    if(selected_gateway != -1):
        start_task_timer(num_messages, selected_gateway, payload_cost, life_time)
    
    if selected_gateway == -1:
        return "Full system"
    else:
        return selected_gateway

# Subscribe to the MQTT Broker
def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}`")
        print()
        selected_gateway = intelligentJointAlgorithm(msg.payload.decode())
        #print(f"Selected Gateway: {selected_gateway}")
        #client.publish("gateway/selection", f"Selected Gateway: {selected_gateway}, remaining power: {gateway_powers}")
        #print (q_table)
        #print(f"remaining power: {gateway_powers}")
        #print()

    client.subscribe(topic)
    #client.subscribe("gateway/status") 
    client.on_message = on_message


# Run the MQTT Client
def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
