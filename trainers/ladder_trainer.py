# === ladder_trainer.py ===
import torch
import random
import os
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend for headless environments
import matplotlib.pyplot as plt
from datetime import datetime

from env.tiny_battlegrounds import TinyBattlegroundsEnv
from agents.mlp_agent import SelfLearningAgent
from agents.transformer_agent import TransformerAgent

# === CONFIGURATION ===
NUM_AGENTS = 16
GAME_SIZE = 8
SAVE_EVERY = 20
EVAL_EVERY = 40
PLOT_EVERY = 50
MAX_GEN = 100
AGENT_TYPE = "transformer"  # or "mlp"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# === AGENT FACTORY ===
def make_agent(i):
    if AGENT_TYPE == "transformer":
        return TransformerAgent(name=f"Transformer_{i}")
    elif AGENT_TYPE == "mlp":
        return SelfLearningAgent(input_size=20, action_size=5, name=f"MLP_{i}")
    else:
        raise ValueError(f"Unsupported AGENT_TYPE: {AGENT_TYPE}")

# === MATCHMAKING ===
def matchmake(agents, game_size):
    sorted_agents = sorted(agents, key=lambda a: a.mmr + random.random())
    games = []
    while len(sorted_agents) >= game_size:
        group = sorted_agents[:game_size]
        for agent in group:
            sorted_agents.remove(agent)
        games.append(group)
    return games

# === GAME SIMULATION ===
def run_game(agent_group):
    env = TinyBattlegroundsEnv(agent_group)
    rewards = env.play_game()
    
    for agent in agent_group:
        try:
            agent.learn(rewards[agent.name])
        except Exception as e:
            print(f"⚠️ Learning failed for {agent.name}: {e}")
            continue
            
    return [a.mmr for a in agent_group]

def evaluate_against_snapshots(agent, snapshot_paths):
    """Evaluates current agent against historical snapshots"""
    opponents = []
    for path in snapshot_paths:
        try:
            if AGENT_TYPE == "transformer":
                clone = TransformerAgent(name=f"Snapshot_{os.path.basename(path)}")
                clone.load(path)
                # Critical: Reset all episodic trackers
                clone.memory = []
                clone.turns_skipped_this_game = 0
                clone.gold_spent_this_game = 0 
                clone.minions_bought_this_game = 0
            else:
                clone = SelfLearningAgent(20, 5, name=f"Snapshot_{os.path.basename(path)}")
                clone.load(path)
                clone.memory = []
            
            opponents.append(clone)
            
        except Exception as e:
            print(f"⚠️ Failed to load snapshot {path}: {e}")
            continue

    # Create evaluation environment
    eval_group = [agent] + opponents
    env = TinyBattlegroundsEnv(eval_group)
    
    # Run game with verbose output for the test agent
    rewards = env.play_game(verbose=True, focus_agent_name=agent.name)
    
    # Print evaluation results
    print(f"\n📊 Evaluation Results for {agent.name}:")
    for a in eval_group:
        print(f"  {a.name}: MMR Δ = {rewards.get(a.name, 0):+.1f} (New MMR: {a.mmr:.1f})")

# === PLOTTING ===
def plot_mmr(agents, history, gen):
    avg = sum(agent.mmr for agent in agents) / len(agents)
    top = max(agent.mmr for agent in agents)
    history.append((gen, avg, top))

    if gen % PLOT_EVERY == 0:
        gens, avgs, tops = zip(*history)
        plt.plot(gens, avgs, label="Avg MMR")
        plt.plot(gens, tops, label="Top MMR")
        plt.xlabel("Generation")
        plt.ylabel("MMR")
        plt.title(f"MMR Over Time ({AGENT_TYPE})")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f"mmr_plot_gen_{gen}_{AGENT_TYPE}.png"))
        plt.close()

# === MAIN TRAINING LOOP ===
def main():
    agents = [make_agent(i) for i in range(NUM_AGENTS)]
    mmr_history = []
    generation = 0

    try:
        while generation < MAX_GEN:
            print(f"--------- Generation: {generation} ---------")
            generation += 1

            games = matchmake(agents, GAME_SIZE)
            for game in games:
                run_game(game)

            if generation % EVAL_EVERY == 0:
                top_agent = max(agents, key=lambda a: a.mmr)
                snap_path = os.path.join(SAVE_DIR, f"Snapshot_gen{generation}.pt")
                top_agent.save(snap_path)
                print(f"📸 Saved snapshot: {snap_path}")

                snapshots = sorted([
                    os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)
                    if f.startswith("Snapshot_gen") and f.endswith(".pt")
                ])[-3:]

                evaluate_against_snapshots(top_agent, snapshots)

            plot_mmr(agents, mmr_history, generation)

            if generation % 10 == 0:
                top_agent = max(agents, key=lambda a: a.mmr)
                print(f"[Gen {generation}] Top Agent: {top_agent.name} | MMR: {top_agent.mmr:.2f}")

    except KeyboardInterrupt:
        print("\n🛑 Training manually interrupted.")

    except Exception as e:
        print(f"\n💥 Exception occurred: {e}")

    finally:
        top_agent = max(agents, key=lambda a: a.mmr)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_DIR, f"BEST_{top_agent.name}_{AGENT_TYPE}_{now}.pt")
        top_agent.save(path)
        print(f"🏁 Saved BEST agent: {top_agent.name} | Final MMR: {top_agent.mmr:.2f} → {path}")

if __name__ == "__main__":
    main()
