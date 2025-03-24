import numpy as np
import pickle
import random
import gym
import os
import math

# Global variables for Q-learning
Q_TABLE = {}
EPSILON = 0.1  # Exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration rate
ALPHA = 0.1    # Learning rate
GAMMA = 0.9    # Discount factor
HAVE_PASSENGER = False  # Track if passenger is in taxi
PREV_ACTION = None  # Track previous action

# Try to load pre-trained Q-table if available
try:
    with open('q_table.pickle', 'rb') as f:
        Q_TABLE = pickle.load(f)
    print(f"Loaded Q-table with {len(Q_TABLE)} states")
except FileNotFoundError:
    print("No pre-trained Q-table found, starting fresh")

def get_direction(taxi_pos, target_pos):
    """
    Calculate direction vector from taxi to target position
    Returns normalized direction (-1, 0, or 1 for each dimension)
    """
    taxi_row, taxi_col = taxi_pos
    target_row, target_col = target_pos
    
    row_diff = target_row - taxi_row
    col_diff = target_col - taxi_col
    
    # Normalize to -1, 0, 1
    row_dir = 0 if row_diff == 0 else (1 if row_diff > 0 else -1)
    col_dir = 0 if col_diff == 0 else (1 if col_diff > 0 else -1)
    
    return row_dir, col_dir

def extract_compact_state(obs):
    """
    Extract a compact state representation from observation
    Instead of raw station positions, use directions to stations
    """
    global HAVE_PASSENGER, PREV_ACTION
    
    # Unpack observation
    taxi_row, taxi_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # Update passenger status based on previous action
    if PREV_ACTION == 4:  # PICKUP
        if passenger_look and (taxi_row, taxi_col) == (s0_row, s0_col) or \
                             (taxi_row, taxi_col) == (s1_row, s1_col) or \
                             (taxi_row, taxi_col) == (s2_row, s2_col) or \
                             (taxi_row, taxi_col) == (s3_row, s3_col):
            HAVE_PASSENGER = True
    elif PREV_ACTION == 5:  # DROPOFF
        HAVE_PASSENGER = False
    
    # Get station positions
    stations = [
        (s0_row, s0_col),
        (s1_row, s1_col),
        (s2_row, s2_col),
        (s3_row, s3_col)
    ]
    
    # Detect passenger and destination stations
    passenger_station = None
    destination_station = None
    
    taxi_pos = (taxi_row, taxi_col)
    
    # Check which station has the passenger
    for station in stations:
        # If passenger is visible and taxi is at a station
        if passenger_look and taxi_pos == station and not HAVE_PASSENGER:
            passenger_station = station
        # If destination is visible and taxi is at a station
        elif destination_look and taxi_pos == station and HAVE_PASSENGER:
            destination_station = station
    
    # If we know where passenger is, get direction to them
    if passenger_station and not HAVE_PASSENGER:
        passenger_dir = get_direction(taxi_pos, passenger_station)
    else:
        passenger_dir = (0, 0)  # No direction if passenger is picked up or unknown
    
    # If we know where destination is, get direction to it
    if destination_station and HAVE_PASSENGER:
        destination_dir = get_direction(taxi_pos, destination_station)
    else:
        destination_dir = (0, 0)  # No direction if destination is unknown or passenger not picked up
    
    # Compact state representation
    compact_state = (
        passenger_dir[0], passenger_dir[1],  # Direction to passenger
        destination_dir[0], destination_dir[1],  # Direction to destination
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,  # Obstacles
        int(HAVE_PASSENGER),  # Whether passenger is in taxi
        int(passenger_look),  # Whether passenger is visible
        int(destination_look)  # Whether destination is visible
    )
    
    return compact_state

def state_to_key(obs):
    """Convert observation to a hashable key for Q-table"""
    compact_state = extract_compact_state(obs)
    return compact_state

def get_action(obs, training=True):
    """
    Select an action based on current observation
    If training is True, use epsilon-greedy policy
    If training is False, use greedy policy
    """
    global PREV_ACTION, EPSILON
    
    state_key = state_to_key(obs)
    
    # If this is a new state, initialize it in Q-table with zeros
    if state_key not in Q_TABLE:
        Q_TABLE[state_key] = np.zeros(6)  # 6 possible actions
    
    # Epsilon-greedy strategy for exploration vs exploitation
    if training and random.random() < EPSILON:
        # Exploration: choose random action
        action = random.choice([0, 1, 2, 3, 4, 5])
    else:
        # Exploitation: choose best action
        action = np.argmax(Q_TABLE[state_key])
    
    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    
    # Update previous action
    PREV_ACTION = action
    
    return action

def update_q_table(state, action, reward, next_state, done):
    """Update Q-table using Q-learning algorithm"""
    state_key = state_to_key(state)
    next_state_key = state_to_key(next_state)
    
    # Initialize new states with zeros
    if state_key not in Q_TABLE:
        Q_TABLE[state_key] = np.zeros(6)
    if next_state_key not in Q_TABLE:
        Q_TABLE[next_state_key] = np.zeros(6)
    
    # Q-learning update rule
    if not done:
        Q_TABLE[state_key][action] += ALPHA * (
            reward + GAMMA * np.max(Q_TABLE[next_state_key]) - Q_TABLE[state_key][action]
        )
    else:
        # Terminal state
        Q_TABLE[state_key][action] += ALPHA * (reward - Q_TABLE[state_key][action])

def save_q_table(filename='q_table.pickle'):
    """Save Q-table to file"""
    with open(filename, 'wb') as f:
        pickle.dump(Q_TABLE, f)
    print(f"Saved Q-table with {len(Q_TABLE)} states")

# Additional training function that can be called externally
def train_agent(env, episodes=1000):
    """Train the agent on the given environment"""
    global HAVE_PASSENGER, PREV_ACTION
    
    for episode in range(episodes):
        state, _ = env.reset()
        HAVE_PASSENGER = False
        PREV_ACTION = None
        done = False
        total_reward = 0
        
        while not done:
            action = get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            save_q_table()
    
    # Save final Q-table
    save_q_table()
    return