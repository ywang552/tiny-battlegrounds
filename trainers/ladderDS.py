# === ladder_trainer.py ===
import torch
import random
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.tiny_battlegrounds import TinyBattlegroundsEnv
from agents.mlp_agent import SelfLearningAgent
from agents.transformer_agent import TransformerAgent

# === CONFIG ===
NUM_CORES = os.cpu_count()
POPULATION_SIZE = 32
GAME_SIZE = 8
MAX_GEN = 500
LEAGUE_SIZE = 8
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# === MATCHMAKING ===
def matchmake(agents, game_size):
    """Create balanced games with random MMR mixing"""
    shuffled = sorted(agents, key=lambda x: x.mmr + random.uniform(-100, 100))
    return [shuffled[i:i+game_size] for i in range(0, len(shuffled), game_size)]

class TrainingLeague:
    def __init__(self, capacity=LEAGUE_SIZE):
        self.capacity = capacity
        self.members = []

    def save_snapshots(self, generation, save_dir):
            """Save all league members with type prefixes"""
            for i, member in enumerate(self.members):
                agent = member['agent']
                save_agent(agent, f"league_gen{generation}_rank{i}", save_dir)

        
    def add_member(self, agent):
        if len(self.members) >= self.capacity:
            self.members.sort(key=lambda x: x['score'])
            self.members.pop(0)
        self.members.append({
            'agent': agent,
            'score': 0,
            'timestamp': datetime.now()
        })
    
    def get_opponents(self, num_opponents):
        candidates = sorted(self.members, 
                          key=lambda x: x['score'] + 0.1*(x['timestamp'].timestamp()),
                          reverse=True)
        return [self.clone_agent(c['agent']) for c in candidates[:num_opponents]]

    def clone_agent(self, agent):
        """Universal cloning method for league agents"""
        if isinstance(agent, TransformerAgent):
            clone = TransformerAgent(name=f"Clone_{agent.name}")
            clone.load_state_dict(agent.state_dict())
            clone.mmr = agent.mmr
            return clone
        elif isinstance(agent, SelfLearningAgent):
            clone = SelfLearningAgent(19, 5, name=f"Clone_{agent.name}")
            clone.policy.load_state_dict(agent.policy.state_dict())
            clone.value_net.load_state_dict(agent.value_net.state_dict())
            clone.mmr = agent.mmr
            return clone

class PopulationManager:
    def __init__(self, pop_size=POPULATION_SIZE):
        # Initialize with higher starting MMR
        self.population = []
        for i in range(pop_size):
            if i < pop_size//2:
                agent = TransformerAgent(name=f"Transformer_{i}")
                agent.mmr = 1000  # Starting MMR
            else:
                agent = SelfLearningAgent(19, 5, name=f"MLP_{i}")
                agent.mmr = 1000  # Starting MMR
            self.population.append(agent)
        self.elite_pool = [self.clone_agent(self.population[0])]


    def clone_agent(self, agent):
        """Population cloning method"""
        if isinstance(agent, TransformerAgent):
            clone = TransformerAgent(name=f"Clone_{agent.name}")
            clone.load_state_dict(agent.state_dict())
            clone.mmr = agent.mmr
            return clone
        elif isinstance(agent, SelfLearningAgent):
            clone = SelfLearningAgent(19, 5, name=f"Clone_{agent.name}")
            clone.policy.load_state_dict(agent.policy.state_dict())
            clone.value_net.load_state_dict(agent.value_net.state_dict())
            clone.mmr = agent.mmr
            return clone

    def mutate_agent(self, agent):
        """Universal mutation method"""
        if isinstance(agent, TransformerAgent):
            with torch.no_grad():
                for param in agent.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
        elif isinstance(agent, SelfLearningAgent):
            with torch.no_grad():
                for param in agent.policy.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
                for param in agent.value_net.parameters():
                    param.add_(torch.randn_like(param) * 0.01)

    def evolve_population(self):
        # Only evolve if we have positive MMR progress
        if any(a.mmr > 10 for a in self.population):

            # Tournament selection
            candidates = random.sample(self.population, k=min(8, len(self.population)))
            candidates.sort(key=lambda a: a.mmr, reverse=True)
            
            new_generation = []
            # Keep top 4 candidates
            for i in range(4):
                if i < len(candidates):
                    new_generation.append(self.clone_agent(candidates[i]))
            
            # Add mutations
            for i in range(2):
                if i < len(new_generation):
                    mutant = self.clone_agent(new_generation[i])
                    self.mutate_agent(mutant)
                    new_generation.append(mutant)
            
            # Preserve elites
            self.population = new_generation + [self.clone_agent(a) for a in self.elite_pool[-2:]]

def create_worker():
    torch.set_num_threads(1)
    random.seed(os.getpid())

def run_parallel_game(agents):
    try:
        env = TinyBattlegroundsEnv(agents)
        game_result = env.play_game()  # Environment just needs to run the game
        
        # Calculate rewards based on MMR changes
        rewards = {}
        for agent in agents:
            # Base reward is just the agent's current MMR
            reward = agent.mmr
            
            # Add bonuses for in-game actions
            if hasattr(agent, 'gold_spent_this_game'):
                reward += agent.gold_spent_this_game * 0.1
                agent.gold_spent_this_game = 0
                
            if hasattr(agent, 'minions_bought_this_game'):
                reward += agent.minions_bought_this_game * 0.5
                agent.minions_bought_this_game = 0
                
            rewards[agent.name] = reward
            
        return rewards
        
    except Exception as e:
        print(f"⚠️ Game failed: {e}")
        # Fallback equal rewards
        return {agent.name: 1000 for agent in agents}  # Default MMR




def save_agent(agent, generation, save_dir):
    """Save agent with type-specific filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type = "Transformer" if isinstance(agent, TransformerAgent) else "MLP"
    filename = f"{agent_type}_gen{generation}_{timestamp}.pt"
    os.makedirs(os.path.join(save_dir, agent_type), exist_ok=True)
    save_path = os.path.join(save_dir, agent_type, filename)
    agent.save(save_path)
    print(f"💾 Saved {agent_type} agent: {filename}")
    return save_path

# === METRICS TRACKING ADDITIONS ===
class TrainingMetrics:
    def __init__(self):
        self.gen_history = []
        self.mmr_history = []
        self.win_rates = []
        self.action_distributions = []
        self.loss_history = []
        self.agent_type_performance = {'Transformer': [], 'MLP': []}

    def update(self, generation, population):
        # Directly use agent MMRs
        trans_mmrs = [a.mmr for a in population if isinstance(a, TransformerAgent)]
        mlp_mmrs = [a.mmr for a in population if isinstance(a, SelfLearningAgent)]
        

        self.gen_history.append(generation)
        self.mmr_history.append((
            np.mean(trans_mmrs + mlp_mmrs) if (trans_mmrs or mlp_mmrs) else 0,
            max(trans_mmrs + mlp_mmrs) if (trans_mmrs or mlp_mmrs) else 0
        ))
        
        # Win rates based on MMR distribution
        self.agent_type_performance['Transformer'].append(
            np.mean(trans_mmrs) / 100 if trans_mmrs else 0
        )
        self.agent_type_performance['MLP'].append(
            np.mean(mlp_mmrs) / 100 if mlp_mmrs else 0
        )


    def plot_progress(self, save_dir):
        if not self.gen_history or not self.mmr_history:
            print("⚠️ Skipping plot: no data yet.")
            return

        plt.figure(figsize=(15, 10))
        
        # MMR Progress
        plt.subplot(2, 2, 1)
        gens, mmrs = zip(*[(g, m[0]) for g, m in zip(self.gen_history, self.mmr_history)])
        plt.plot(gens, mmrs, label='Avg MMR')
        gens, max_mmrs = zip(*[(g, m[1]) for g, m in zip(self.gen_history, self.mmr_history)])
        plt.plot(gens, max_mmrs, label='Max MMR')
        plt.title("MMR Progress")
        plt.xlabel("Generation")
        plt.ylabel("MMR")
        plt.legend()
        plt.grid()

        # Agent Type Performance
        plt.subplot(2, 2, 2)
        plt.plot(self.gen_history, self.agent_type_performance['Transformer'], label='Transformer')
        plt.plot(self.gen_history, self.agent_type_performance['MLP'], label='MLP')
        plt.title("Win Rate by Agent Type")
        plt.xlabel("Generation")
        plt.ylabel("Avg Wins")
        plt.legend()
        plt.grid()

        # Action Distribution
        if self.action_distributions:
            plt.subplot(2, 2, 3)
            actions = ['buy', 'sell', 'roll', 'level', 'end_turn']
            for i, action in enumerate(actions):
                plt.plot(self.gen_history, [d[i] for d in self.action_distributions], label=action)
            plt.title("Action Distribution")
            plt.xlabel("Generation")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()

        # Loss History
        if self.loss_history:
            plt.subplot(2, 2, 4)
            plt.plot(self.gen_history, self.loss_history)
            plt.title("Training Loss")
            plt.xlabel("Generation")
            plt.ylabel("Loss")
            plt.grid()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"training_metrics_gen_{self.gen_history[-1]}.png"))
        plt.close()

# === MODIFIED MAIN FUNCTION ===
def main():
    metrics = TrainingMetrics()
    league = TrainingLeague()
    population = PopulationManager()
    executor = ProcessPoolExecutor(max_workers=NUM_CORES, initializer=create_worker)
    
    try:
        for gen in range(MAX_GEN):
            print(f"\n=== Generation {gen} ===")
            
            # Maintain population size
            while len(population.population) < GAME_SIZE:
                population.population.append(population.clone_agent(random.choice(population.population)))
            
            # Run parallel games
            games = matchmake(population.population, GAME_SIZE)
            futures = [executor.submit(run_parallel_game, game) for game in games]
            
            # Process results and collect metrics
            game_results = []
            for future in futures:
                rewards = future.result()
                game_results.append(rewards)
                for agent in population.population:
                    if agent.name in rewards:
                        agent.learn(rewards[agent.name])
            
            # Calculate and log metrics
            metrics.update(gen, population.population)

            
            # Evolutionary steps
            population.evolve_population()
            population.elite_pool.append(max(population.population, key=lambda a: a.mmr))
            
            # Print generation summary
            print(f"Generation {gen} Summary:")
            print(f"  Avg MMR: {metrics.mmr_history[-1][0]:.1f}")
            print(f"  Max MMR: {metrics.mmr_history[-1][1]:.1f}")
            print(f"  Transformer Win Rate: {metrics.agent_type_performance['Transformer'][-1]:.2f}")
            print(f"  MLP Win Rate: {metrics.agent_type_performance['MLP'][-1]:.2f}")
            
            # Save checkpoints and plots
            if gen % 10 == 0:
                champion = population.elite_pool[-1]
                save_agent(champion, gen, SAVE_DIR)
                
                # Save sample of population
                for agent in random.sample(population.population, min(3, len(population.population))):
                    save_agent(agent, gen, SAVE_DIR)


                metrics.plot_progress(SAVE_DIR)
                
    finally:
        # Final save
        if population.elite_pool:
            best_agent = max(population.elite_pool, key=lambda a: a.mmr)
        else:
            best_agent = max(population.population, key=lambda a: a.mmr)
        save_agent(best_agent, "final", SAVE_DIR)
        # Save all elites
        for i, elite in enumerate(population.elite_pool):
            save_agent(elite, f"elite{i}", SAVE_DIR)

        metrics.plot_progress(SAVE_DIR)
        executor.shutdown()

if __name__ == "__main__":
    main()