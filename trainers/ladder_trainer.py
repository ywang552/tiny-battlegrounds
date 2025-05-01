# === ladder_trainer.py ===
import torch
import random
import time
import os
import signal
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from env.tiny_battlegrounds import TinyBattlegroundsEnv
from agents.mlp_agent import SelfLearningAgent

# === CONFIGURATION ===
NUM_AGENTS = 64
GAME_SIZE = 8
SAVE_EVERY = 20
EVAL_EVERY = 40
MAX_WORKERS = 4
PLOT_EVERY = 50

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

global_agents = []  # for graceful interrupt access

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

# === INTERRUPT HANDLING ===
def save_all_agents(tag="INTERRUPT"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for agent in global_agents:
        path = os.path.join(SAVE_DIR, f"agent_{agent.name}_{tag}_{timestamp}.pt")
        agent.save(path)
    print(f"\n✅ Saved {len(global_agents)} agents to {SAVE_DIR}/")

def handle_interrupt(signum, frame):
    print("\n🚨 Interrupt received. Saving all agents...")
    save_all_agents()
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

# === GAME SIMULATION ===
def run_game(agent_group):
    env = TinyBattlegroundsEnv(agent_group)
    rewards = env.play_game()
    for agent in agent_group:
        agent.learn(rewards[agent.name])
    return [a.mmr for a in agent_group]

# === SNAPSHOT EVALUATION ===
def evaluate_against_snapshots(agent, snapshot_paths):
    opponents = []
    for path in snapshot_paths:
        clone = SelfLearningAgent(9, 5, name=f"Snapshot_{os.path.basename(path)}")
        clone.load(path)
        opponents.append(clone)

    eval_group = opponents + [agent]
    env = TinyBattlegroundsEnv(eval_group)
    env.play_game(verbose=False)
    print(f"\n📊 Evaluation of {agent.name} against {[op.name for op in opponents]} done.")

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
        plt.title("MMR Over Time")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f"mmr_plot_gen_{gen}.png"))
        plt.close()

# === MAIN TRAINING LOOP ===
def main():
    global global_agents
    global_agents = [SelfLearningAgent(9, 5, name=f"Agent_{i}") for i in range(NUM_AGENTS)]

    mmr_history = []
    generation = 0

    try:
        while True:
            generation += 1
            games = matchmake(global_agents, GAME_SIZE)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                executor.map(run_game, games)

            if generation % SAVE_EVERY == 0:
                save_all_agents(tag=f"gen{generation}")

            if generation % EVAL_EVERY == 0:
                top_agent = max(global_agents, key=lambda a: a.mmr)
                snapshots = sorted([
                    os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)
                    if f.endswith(".pt") and "Snapshot" not in f
                ])[-3:]  # last 3 snapshots
                evaluate_against_snapshots(top_agent, snapshots)

            plot_mmr(global_agents, mmr_history, generation)

            if generation % 10 == 0:
                top_agent = max(global_agents, key=lambda a: a.mmr)
                print(f"[Gen {generation}] Top Agent: {top_agent.name} | MMR: {top_agent.mmr:.2f}")

    except Exception as e:
        print(f"\n💥 Exception occurred: {e}")
        save_all_agents(tag="CRASH")

if __name__ == "__main__":
    main()
