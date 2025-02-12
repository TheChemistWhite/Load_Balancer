import json
from turtle import distance, update
import pandas as pd
import math
import numpy as np
from collections import defaultdict
import threading
import pickle
import os
import glob
from requests import get
from score_logger import ScoreLogger
from statistics import mean 

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

class DefaultDict(defaultdict):
        def __missing__(self, key):
            self[key] = defaultdict(float)
            return self[key]

def save_variable(table, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(table, f)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def load_variable(filename):
    try:
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            return DefaultDict(defaultdict)
            
        if os.path.getsize(filename) == 0:
            print(f"File {filename} is empty")
            return DefaultDict(defaultdict)
            
        with open(filename, 'rb') as f:
            print(f"Loading Q-table from {filename}")
            data = pickle.load(f)
            return data
            
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error reading {filename}, creating new Q-table")
        return DefaultDict(defaultdict)
    except Exception as e:
        print(f"Unexpected error loading {filename}: {e}")
        return DefaultDict(defaultdict)

print("\nInitializing Q-table...")
if os.path.exists('q_table.pkl'):
    q_table = load_variable('q_table.pkl')
else:
    print("Creating new Q-table")
    q_table = DefaultDict(defaultdict)
    save_variable(q_table, 'q_table.pkl')

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

#Load the snapshots
snapshots = glob.glob(f'{dataset_folder}/*.csv')

# Load the list of gateways
with open(gateways_list_file, 'r') as f:
    gateways = [line.strip() for line in f]

gateways_positions = pd.read_csv(gateways_positions_file)

active_tasks = {}

def reset_payload_cost(gateway_idx, payload_cost, mex_idx):
    global gateway_powers
    
    gateway_powers[gateway_idx] += payload_cost
    if gateway_powers[gateway_idx] > GWResource:
        gateway_powers[gateway_idx] = GWResource  
    #print(f"Task {mex_idx} completed for Gateway {gateway_idx}. Reset power by {payload_cost}. Current power: {gateway_powers[gateway_idx]}")

    # Safely remove the task from active_tasks
    if mex_idx in active_tasks:
        del active_tasks[mex_idx]
    else:
        print(f"Warning: message {mex_idx} not found in active tasks.")


def start_task_timer(mex_idx, gateway_idx, payload_cost, time):
    timer = threading.Timer(time, reset_payload_cost, args=(gateway_idx, payload_cost, mex_idx))
    timer.start()
    active_tasks[mex_idx] = {"gateway": gateway_idx, "payload_cost": payload_cost, "timer": timer}

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

def get_best_action(state, possible_actions):
   
    q_values = [q_table[state][action] for action in possible_actions]
    max_q = max(q_values)
    best_actions = [action for action, q_value in zip(possible_actions, q_values) 
                    if q_value == max_q]
    return np.random.choice(best_actions)

def get_best_near_gw(near_gw):

    gw = near_gw[0]
    for i in near_gw:
        j = i + 1
        for j in near_gw:
            if i != j:
                if gateway_powers[i] > gateway_powers[j]:
                    #print ("gw:", i, "power:", gateway_powers[i], "gw:", j, "power:", gateway_powers[j])
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

def calculate_reward(gateway_idx, near_gw, operation_cost):

    remaining_power = gateway_powers[gateway_idx] - operation_cost
    #print (gateway_idx)

    if(gateway_idx in near_gw):
        return  1
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

score_log = ScoreLogger("Random", 1000)

def intelligentJointAlgorithm(msg):
    message = json.loads(msg)
    payload_cost = 5 
    sf = message['spreading_factor']
    global num_messages
    num_messages += 1
    distances = tuple(get_distances(message['x_coordinate'], message['y_coordinate']))

    rx_gw = message['Receiving gateways']

    near_gw = tuple(i for i in range(number_of_gateways) if gateways[i] in rx_gw)

    current_state = get_state(near_gw, payload_cost)
    possible_actions = get_possible_actions(payload_cost)
    selected_gateway = get_action(current_state, possible_actions)

    if selected_gateway == -1:
        assigned = 0
        reward = -5
        TOTAL_COST = 50
        gw_power = 0
    else:
        assigned = 1
        gw_power = gateway_powers[selected_gateway]
        update_gateway_power(selected_gateway, payload_cost)
        reward = calculate_reward(selected_gateway, near_gw, payload_cost)
        if(selected_gateway in near_gw):
            TOTAL_COST = payload_cost
        else:
            TOTAL_COST = payload_cost + distances[selected_gateway]
        
    #print(f"Total cost: {TOTAL_COST}, selected gateway: {selected_gateway}, reward: {reward}")

    score_log.add_score(int(TOTAL_COST), distances[selected_gateway], assigned, gw_power, reward, gateway_powers, num_messages)

    next_state = get_state(near_gw, payload_cost)
    next_possible_actions = get_possible_actions(payload_cost)
    update(current_state, selected_gateway, reward, next_state, next_possible_actions)

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
    
def Random(msg):
    message = json.loads(msg)
    payload_cost = 5 
    sf = message['spreading_factor']
    global num_messages
    num_messages += 1
    distances = tuple(get_distances(message['x_coordinate'], message['y_coordinate']))

    rx_gw = message['Receiving gateways']

    near_gw = tuple(i for i in range(number_of_gateways) if gateways[i] in rx_gw)

    selected_gateway = choose_random_gw(payload_cost)

    if selected_gateway == -1:
        assigned = 0
        reward = -5
        TOTAL_COST = 50
        gw_power = 0
    else:
        assigned = 1
        gw_power = gateway_powers[selected_gateway]
        update_gateway_power(selected_gateway, payload_cost)
        reward = calculate_reward(selected_gateway, near_gw, payload_cost)
        if(selected_gateway in near_gw):
            TOTAL_COST = payload_cost
        else:
            TOTAL_COST = payload_cost + distances[selected_gateway]

    score_log.add_score(int(TOTAL_COST), distances[selected_gateway], assigned, gw_power, reward, gateway_powers, num_messages)
    
    life_time = (sf + payload_cost)

    if(selected_gateway != -1):
        start_task_timer(num_messages, selected_gateway, payload_cost, life_time)
    
    if selected_gateway == -1:
        return "Full system"
    else:
        return selected_gateway


if __name__ == '__main__':

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
                
            message = {
                    "x_coordinate": row['x'],
                    "y_coordinate": row['y'],
                    "spreading_factor": row['spreading_factor'],
                    "Receiving gateways" : RX_GW
                }
            
            message = json.dumps(message)
            #gw = intelligentJointAlgorithm(message)
            gw = Random(message)
            #print(f"Message: {num_messages} sent to gateway: {gw}")

    print("End of simulation")
    save_variable(q_table, 'q_table.pkl')
    print(f"Saved Q-table")
