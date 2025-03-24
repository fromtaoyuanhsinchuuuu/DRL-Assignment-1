import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv
import student_agent  # Import your agent

def train_agent(episodes=5000, render_every=1000):
    """Train the agent and save Q-table to a pickle file"""
    # Initialize environment with a reasonable fuel limit for training
    env = SimpleTaxiEnv(fuel_limit=5000)
    
    # Training parameters
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    
    # Set initial epsilon
    student_agent.EPSILON = epsilon_start
    
    # Training statistics
    rewards_history = []
    steps_history = []
    
    print("Starting training...")
    for episode in range(1, episodes+1):
        # Reset environment at start of each episode
        obs, _ = env.reset()
        
        # Reset passenger status at the beginning of each episode
        student_agent.HAVE_PASSENGER = False
        student_agent.PREV_ACTION = None
        
        total_reward = 0
        done = False
        steps = 0
        
        # Decay epsilon
        student_agent.EPSILON = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        
        # Run one episode
        while not done:
            # Select action using epsilon-greedy policy
            action = student_agent.get_action(obs, training=True)
            
            # Execute action
            next_obs, reward, done, _ = env.step(action)
            
            # Update Q-table
            student_agent.update_q_table(obs, action, reward, next_obs, done)
            
            # Update state and rewards
            obs = next_obs
            total_reward += reward
            steps += 1
            
            # Render environment occasionally
            if episode % render_every == 0 and steps < 50:  # Only render first 50 steps to avoid lengthy visualizations
                taxi_row, taxi_col = obs[0], obs[1]
                env.render_env((taxi_row, taxi_col), action=action, step=steps, fuel=env.current_fuel)
                time.sleep(0.05)  # Small delay for visualization
        
        # Record episode statistics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Display training progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            avg_steps = np.mean(steps_history[-100:]) if len(steps_history) >= 100 else np.mean(steps_history)
            
            print(f"Episode {episode}/{episodes}, " +
                  f"Avg Reward: {avg_reward:.2f}, " +
                  f"Avg Steps: {avg_steps:.2f}, " +
                  f"Epsilon: {student_agent.EPSILON:.3f}, " +
                  f"Q-table size: {len(student_agent.Q_TABLE)}")
        
    # Save final Q-table
    student_agent.save_q_table('q_table.pickle')
    
    print("Training completed!")
    print(f"Final Q-table size: {len(student_agent.Q_TABLE)}")
    
    # Plot training curves
    try:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot rewards
        ax1.plot(rewards_history)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot moving average of rewards
        window_size = min(100, len(rewards_history))
        if window_size > 0:
            moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, window_size-1+len(moving_avg)), moving_avg, 'r-', linewidth=2)
            ax1.legend(['Rewards', f'{window_size}-episode Moving Average'])
        
        # Plot steps per episode
        ax2.plot(steps_history)
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        
        # Plot moving average of steps
        if window_size > 0:
            steps_moving_avg = np.convolve(steps_history, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, window_size-1+len(steps_moving_avg)), steps_moving_avg, 'r-', linewidth=2)
            ax2.legend(['Steps', f'{window_size}-episode Moving Average'])
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("Saved training metrics plot to training_metrics.png")
    except ImportError:
        print("Could not plot training metrics (matplotlib required)")
    
    return student_agent.Q_TABLE

if __name__ == "__main__":
    # Train the agent
    q_table = train_agent(episodes=50000)
    
    # Display a sample of the Q-table
    print("\nQ-table Sample (first 5 states):")
    count = 0
    for state, actions in q_table.items():
        print(f"State: {state}")
        print(f"Actions: {actions}")
        count += 1
        if count >= 5:
            break