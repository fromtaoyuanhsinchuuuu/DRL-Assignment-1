import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv
import student_agent  # Import your agent
import random

# Global variables for Q-learning
Q_TABLE = {}
EPSILON = 0.1  # Exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration rate
ALPHA = 0.1    # Learning rate
GAMMA = 0.9    # Discount factor
HAVE_PASSENGER = False  # Track if passenger is in taxi
PREV_ACTION = None  # Track previous action

# Action bias parameters
MOVEMENT_BIAS = 0.5  # Positive bias for movement actions
OPERATION_BIAS = -1.0  # Negative bias for pickup/dropoff operations

# Try to load pre-trained Q-table if available
try:
    with open('q_table.pickle', 'rb') as f:
        Q_TABLE = pickle.load(f, encoding='latin1', fix_imports=True)
    print(f"Loaded Q-table with {len(Q_TABLE)} states")
except FileNotFoundError:
    print("No pre-trained Q-table found, starting fresh")
except Exception as e:
    print(f"Error loading Q-table: {e}")
    Q_TABLE = {}

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
        # Initialize with bias towards movement actions
        for i in range(4):  # Movement actions (0,1,2,3)
            Q_TABLE[state_key][i] = MOVEMENT_BIAS
        for i in range(4, 6):  # Pickup/dropoff actions (4,5)
            Q_TABLE[state_key][i] = OPERATION_BIAS
    
    # Epsilon-greedy strategy for exploration vs exploitation
    if training and random.random() < EPSILON:
        # Exploration: choose random action with bias toward movement
        if random.random() < 0.8:  # 80% chance to select movement action during exploration
            action = random.choice([0, 1, 2, 3])
        else:
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
    """
    Update Q-table using Q-learning algorithm
    With additional bias to prefer movement actions
    """
    state_key = state_to_key(state)
    next_state_key = state_to_key(next_state)
    
    # Initialize new states with zeros and action biases
    if state_key not in Q_TABLE:
        Q_TABLE[state_key] = np.zeros(6)
        # Initialize with bias towards movement actions
        for i in range(4):  # Movement actions (0,1,2,3)
            Q_TABLE[state_key][i] = MOVEMENT_BIAS
        for i in range(4, 6):  # Pickup/dropoff actions (4,5)
            Q_TABLE[state_key][i] = OPERATION_BIAS
            
    if next_state_key not in Q_TABLE:
        Q_TABLE[next_state_key] = np.zeros(6)
        # Initialize with bias towards movement actions
        for i in range(4):  # Movement actions (0,1,2,3)
            Q_TABLE[next_state_key][i] = MOVEMENT_BIAS
        for i in range(4, 6):  # Pickup/dropoff actions (4,5)
            Q_TABLE[next_state_key][i] = OPERATION_BIAS
    
    # Adjust reward based on action type to encourage movement actions
    adjusted_reward = reward
    if action < 4:  # Movement actions
        # Slight boost to movement actions
        adjusted_reward += 0.1
    else:  # Pickup/dropoff actions
        # Only perform pickup/dropoff when really necessary
        # If reward is negative (wrong action), make it even more negative
        if reward < 0:
            adjusted_reward *= 1.5  # Amplify negative rewards for wrong pickup/dropoff
    
    # Q-learning update rule with adjusted reward
    if not done:
        Q_TABLE[state_key][action] += ALPHA * (
            adjusted_reward + GAMMA * np.max(Q_TABLE[next_state_key]) - Q_TABLE[state_key][action]
        )
    else:
        # Terminal state
        Q_TABLE[state_key][action] += ALPHA * (adjusted_reward - Q_TABLE[state_key][action])

def save_q_table(filename='q_table.pickle'):
    """Save Q-table to file"""
    with open(filename, 'wb') as f:
        pickle.dump(Q_TABLE, f)
    print(f"Saved Q-table with {len(Q_TABLE)} states")

def train_agent(env, episodes=2000):
    """Train the agent on the given environment with bias toward movement actions"""
    global HAVE_PASSENGER, PREV_ACTION
    
    best_reward = -float('inf')
    
    for episode in range(episodes):
        state, _ = env.reset()
        HAVE_PASSENGER = False
        PREV_ACTION = None
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Optional: Limit maximum steps per episode to prevent infinite loops
            if steps > 200:
                done = True
        
        # Print progress and save periodically
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}")
            save_q_table()
        
        # Save the best model so far
        if total_reward > best_reward:
            best_reward = total_reward
            save_q_table('best_q_table.pickle')
    
    # Save final Q-table
    save_q_table()
    print(f"Training completed. Best reward: {best_reward:.2f}")
    return

if __name__ == "__main__":
    # Import the environment
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Create the environment
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=200)
    
    # Train the agent
    print("Starting training...")
    start_time = time.time()
    train_agent(env, episodes=2000)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Test the trained agent
    state, _ = env.reset()
    HAVE_PASSENGER = False
    PREV_ACTION = None
    done = False
    total_reward = 0
    steps = 0
    
    print("\nTesting trained agent...")
    while not done and steps < 100:
        action = get_action(state, training=False)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        # Render the environment (if SimpleTaxiEnv supports it)
        taxi_row, taxi_col = state[0], state[1]
        env.render_env((taxi_row, taxi_col), action=action, step=steps, fuel=env.current_fuel)
        time.sleep(0.5)
    
    print(f"Test completed. Total Reward: {total_reward:.2f}, Steps: {steps}")