import random
import torch
from models.tiny_nn_policy import SelfLearningAgent  # <-- import your self-learning agent!
import matplotlib.pyplot as plt  # Add this at the top of your file if not imported yet
import os
import json
from load import load_minions

with open("data/minion_pool.json", "r") as f:
    minion_data = json.load(f)

class Minion:
    def __init__(self, name, types, attack, health, tier, keywords=None):
        self.name = name
        self.types = types
        self.attack = attack
        self.health = health
        self.tier = tier
        self.keywords = set(keywords or [])

    def strength(self):
        return self.attack + self.health

    def __repr__(self):
        return f"{self.name}({self.attack}/{self.health})"

path = "data\\bg_minions_all.json"  # your full file
MINION_POOL = [Minion(**data) for data in load_minions(path)]


# --- TinyBattlegrounds Environment ---
class TinyBattlegroundsEnv:
    def __init__(self, agents):
        self.agents = agents
        self.turn = 1
        self.dead = {}  # {agent: turn number}
        self.latest_dead_agent = None


    def get_base_upgrade_cost(self, tier):
        upgrade_table = {
            1: 5,
            2: 7,
            3: 8,
            4: 9,
            5: 10,
        }
        return upgrade_table.get(tier, float('inf'))

    def get_upgrade_cost(self, agent):
        return agent.tavern_upgrade_cost  # ✅ Use this instead of re-computing


    def setup(self):
        self.turn = 1
        self.dead = {}
        self.latest_dead_agent = None

        for agent in self.agents:
            agent.gold_cap = 3
            agent.gold = 3
            agent.tier = 1
            agent.board = []
            agent.health = 40
            agent.alive = True
            agent.shop = self.roll_shop(agent.tier)
            agent.cumulative_reward = 0  # ← RESET cumulative reward each game!
            agent.gold_spent_this_game = 0
            agent.minions_bought_this_game = 0
            agent.turns_skipped_this_game = 0
            agent.behavior_counts = {'buy': 0, 'sell': 0, 'roll': 0, 'level': 0, 'end_turn': 0}
            agent.last_cumulative_reward = 0
            agent.tavern_upgrade_cost = self.get_base_upgrade_cost(agent.tier)  # ✅ One-time init




    def get_ghost_board(self):
        if not self.dead:
            return []
        most_recent_dead_agent = list(self.dead.keys())[-1]
        return most_recent_dead_agent.board


    def roll_shop(self, tier):
        pool = [m for m in MINION_POOL if m.tier <= tier]

        if tier == 1:
            slots = 3
        elif tier in [2, 3]:
            slots = 4
        elif tier in [4, 5]:
            slots = 5
        elif tier == 6:
            slots = 6
        else:
            slots = 3  # default safety
        
        return random.sample(pool, min(slots, len(pool)))


    def get_available_actions(self, agent):
        actions = []
        upgrade_cost = self.get_upgrade_cost(agent)

        if agent.tier < 6 and agent.gold >= upgrade_cost:
            actions.append("level")


        if agent.shop:
            for idx, minion in enumerate(agent.shop):
                if agent.gold >= 3 and len(agent.board) < 7:
                    actions.append(f"buy_{idx}")

        if agent.gold >= 1:
            actions.append("roll")

        if agent.board:
            for idx, minion in enumerate(agent.board):
                actions.append(f"sell_{idx}")

        # 🔥 Always allow ending turn
        actions.append("end_turn")

        return actions


    def matchmaking(self, alive_agents):
        pairs = []
        random.shuffle(alive_agents)
        if len(alive_agents) % 2 == 1:
            ghost_fighter = alive_agents.pop()
            pairs.append((ghost_fighter, 'ghost'))
        for i in range(0, len(alive_agents), 2):
            pairs.append((alive_agents[i], alive_agents[i+1]))
        return pairs


    def simulate_combat(self, attacker, defender):
        attacker_strength = sum(m.strength() for m in attacker.board)

        if defender == 'ghost':
            if self.latest_dead_agent:
                defender_strength = sum(m.strength() for m in self.latest_dead_agent.board)
                defender_alive_minions = self.latest_dead_agent.board
                defender_tier = self.latest_dead_agent.tier
                defender_real = False
            else:
                defender_strength = 0
                defender_alive_minions = []
                defender_tier = 1
                defender_real = False
        else:
            defender_strength = sum(m.strength() for m in defender.board)
            defender_alive_minions = defender.board
            defender_tier = defender.tier
            defender_real = True

        # 🚨 New rule: punish empty boards
        if len(attacker.board) == 0:
            attacker.health -= 5
            attacker.cumulative_reward -= 5.0  # Optional: punish reward also

        if defender_real and len(defender.board) == 0:
            defender.health -= 5
            defender.cumulative_reward -= 5.0  # Optional: punish reward also

        # ⚔️ Normal fight resolution
        if attacker_strength > defender_strength:
            attacker.cumulative_reward += 4.0
            if defender_real:
                defender.cumulative_reward -= 1.0
                damage = attacker.tier + sum(m.tier for m in attacker.board)
                defender.health -= max(damage, 1)

        elif attacker_strength < defender_strength:
            attacker.cumulative_reward -= 1.0
            if defender_real:
                defender.cumulative_reward += 4.0
                damage = defender_tier + sum(m.tier for m in defender_alive_minions)
                attacker.health -= max(damage, 1)

        else:
            attacker.cumulative_reward += 0.5
            if defender_real:
                defender.cumulative_reward += 0.5
            # No health loss on tie


    def remove_dead(self):
        for agent in self.agents:
            if agent.alive and agent.health <= 0:
                agent.health = max(agent.health, 0)  # ← Cap it!
                agent.alive = False
                self.dead[agent] = self.turn




    def build_state_for_agent(self, agent):
        gold = agent.gold
        tier = agent.tier
        board_strength = sum(m.strength() for m in agent.board)
        num_minions = len(agent.board)
        
        if agent.shop:
            shop_avg_tier = sum(m.tier for m in agent.shop) / len(agent.shop)
        else:
            shop_avg_tier = 0

        shop_slot_0 = agent.shop[0].tier if len(agent.shop) > 0 else 0
        shop_slot_1 = agent.shop[1].tier if len(agent.shop) > 1 else 0
        shop_slot_2 = agent.shop[2].tier if len(agent.shop) > 2 else 0

        turn_number = self.turn

        state_vector = [
            gold, tier, board_strength, num_minions,
            shop_avg_tier, shop_slot_0, shop_slot_1, shop_slot_2,
            turn_number
        ]

        return torch.tensor(state_vector, dtype=torch.float32)



    def step(self, verbose=False, focus_agent_name=None):
        
        for agent in self.agents:
            if not agent.alive:
                continue
            agent.shop = self.roll_shop(agent.tier)
            agent.gold_cap = min(3 + self.turn - 1, 10)
            agent.gold = agent.gold_cap

            if verbose and agent.name == focus_agent_name:
                print(f"\n--- Turn {self.turn} ---")
                print(f"Gold: {agent.gold}, Health: {agent.health}, Tier: {agent.tier}")
                if agent.board:
                    print("Board:")
                    for m in agent.board:
                        print(f"  {m.name} ({m.attack}/{m.health})")
                else:
                    print("Board: EMPTY")
                if agent.shop:
                    print(f"Shop of length {len(agent.shop)}:")
                    for idx, m in enumerate(agent.shop):
                        print(f"  [{idx}] {m.name} ({m.attack}/{m.health})")
                else:
                    print("Shop: EMPTY")

            while agent.gold > 0:
                state = self.build_state_for_agent(agent)
                available_actions = self.get_available_actions(agent)

                if not available_actions:
                    break

                action_idx = agent.act(state)
                action_str = available_actions[action_idx % len(available_actions)]

                reward = agent.cumulative_reward - agent.last_cumulative_reward
                next_state = self.build_state_for_agent(agent)
                if hasattr(agent, "observe"):
                    agent.observe(next_state, reward)


                if hasattr(agent, "behavior_counts"):
                    for key in agent.behavior_counts:
                        if action_str.startswith(key):
                            agent.behavior_counts[key] += 1

                if verbose and agent.name == focus_agent_name:
                    print(f"Action chosen: {action_str}")

                if action_str == "end_turn":
                    if agent.gold > 5:
                        agent.turns_skipped_this_game += 1
                    if verbose and agent.name == focus_agent_name:
                        print(">> Ending turn early.")
                    break

                if action_str == "level":
                    cost = agent.tavern_upgrade_cost

                    if agent.gold >= cost:
                        agent.gold -= cost
                        agent.tier += 1
                        agent.tavern_upgrade_cost = self.get_base_upgrade_cost(agent.tier)  # reset to new tier base
                        agent.cumulative_reward += 2.0
                        agent.gold_spent_this_game += cost
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Upgraded to Tier {agent.tier}.")
                    continue


                if action_str.startswith("buy_"):
                    idx = int(action_str.split("_")[1])
                    if idx < len(agent.shop) and agent.gold >= 3 and len(agent.board) < 7:
                        bought_minion = agent.shop[idx]
                        agent.gold -= 3
                        agent.board.append(agent.shop.pop(idx))
                        agent.cumulative_reward += 1.0
                        agent.minions_bought_this_game += 1
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Bought {bought_minion.name} ({bought_minion.attack}/{bought_minion.health})")
                    continue

                if action_str == "roll":
                    if agent.gold >= 1:
                        agent.gold -= 1
                        agent.shop = self.roll_shop(agent.tier)
                        agent.gold_spent_this_game += 1
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Rolled new shop.")
                    continue

                if action_str.startswith("sell_"):
                    idx = int(action_str.split("_")[1])
                    if idx < len(agent.board):
                        sold_minion = agent.board.pop(idx)
                        agent.gold += 1
                        agent.cumulative_reward += 0.5
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Sold {sold_minion.name} ({sold_minion.attack}/{sold_minion.health})")
                    continue

        
        for agent in self.agents:
            old_cost = agent.tavern_upgrade_cost
            agent.tavern_upgrade_cost = max(agent.tavern_upgrade_cost - 1, 0)
            if verbose and agent.name == focus_agent_name:
                print(f"[Turn {self.turn}] {agent.name} (Tier {agent.tier}) upgrade cost: {old_cost} → {agent.tavern_upgrade_cost}")


        # ⚔️ --- COMBAT PHASE ---
        alive_agents = [a for a in self.agents if a.alive]
        pairs = self.matchmaking(alive_agents)

        for p1, p2 in pairs:
            self.simulate_combat(p1, p2)

        self.remove_dead()






    def play_game(self, verbose=False, focus_agent_name=None):
        self.setup()

        while sum(1 for agent in self.agents if agent.alive) > 1 and self.turn <= 100:
            self.step(verbose=verbose, focus_agent_name=focus_agent_name)
            self.turn += 1

        # ✅ After game ends, fix final survivors who never died
        for agent in self.agents:
            if agent.alive and agent not in self.dead:
                self.dead[agent] = self.turn

            # ✅ Clamp MMR to be non-negative
            agent.mmr = max(0, agent.mmr)

        return self.calculate_rewards()



    def calculate_rewards(self):
        death_turns = sorted(self.dead.items(), key=lambda x: (x[1], random.random()))
        placements = []

        idx = 0
        while idx < len(death_turns):
            same_turn = [death_turns[idx]]
            idx += 1
            while idx < len(death_turns) and death_turns[idx][1] == same_turn[0][1]:
                same_turn.append(death_turns[idx])
                idx += 1
            placements.append(same_turn)

        placement_rewards = {
            1: 80,
            2: 60,
            3: 40,
            4: 20,
            5: -20,
            6: -40,
            7: -60,
            8: -80,
        }

        rewards = {}
        current_place = len(self.agents)

        for tied_group in placements:
            tied_size = len(tied_group)
            reward_sum = sum(placement_rewards.get(current_place - i, -80) for i in range(tied_size))
            average_reward = reward_sum / tied_size
            for agent, _ in tied_group:
                rewards[agent.name] = average_reward
                agent.mmr += average_reward  # ✅ Apply reward
                agent.mmr = max(0, agent.mmr)  # ✅ Clamp here after adding

            current_place -= tied_size

        return rewards



def batch_train(num_batches=100, games_per_batch=20, print_every=10, save_every=5):
    os.makedirs("saved_models", exist_ok=True)  # 🧠 Make sure the folder exists

    agents = [SelfLearningAgent(input_size=9, action_size=5, name=f"Bot_{i}") for i in range(8)]
    for agent in agents:
        agent.mmr = 0  # MMR initialized

    avg_mmr_history = []
    max_mmr_history = []
    env = TinyBattlegroundsEnv(agents)

    try:
        for batch in range(num_batches):
            for game in range(games_per_batch):
                env.setup()
                rewards = env.play_game()

                winner = max(rewards.items(), key=lambda x: x[1])[0]
                # print(f"Game {game+1}: Winner -> {winner}")

                for agent in agents:
                    agent.learn(rewards[agent.name])
                    agent.mmr += rewards[agent.name]

            avg_mmr = sum(agent.mmr for agent in agents) / len(agents)
            max_mmr = max(agent.mmr for agent in agents)

            avg_mmr_history.append(avg_mmr)
            max_mmr_history.append(max_mmr)

            if (batch + 1) % print_every == 0:
                print(f"Batch {batch + 1}: Average MMR = {avg_mmr:.0f}, Highest MMR = {max_mmr:.0f}")

            # 🧠 Save top agent model every `save_every` batches
            if (batch + 1) % save_every == 0:
                top_agent = max(agents, key=lambda a: a.mmr)
                save_data = {
                    "agent_name": top_agent.name,
                    "batch": batch + 1,
                    "mmr": top_agent.mmr,
                    "policy_state_dict": top_agent.policy.state_dict(),
                }
                torch.save(save_data, f"saved_models/agent_{top_agent.name}_batch{batch+1}.pt")
                print(f"✅ Saved {top_agent.name} after Batch {batch+1}")

    except KeyboardInterrupt:
        print("\n🚨 Training interrupted by user! Saving latest models...")
        for agent in agents:
            save_data = {
                "agent_name": agent.name,
                "batch": batch + 1,
                "mmr": agent.mmr,
                "policy_state_dict": agent.policy.state_dict(),
            }
            torch.save(save_data, f"saved_models/agent_{agent.name}_batch{batch+1}_INTERRUPTED.pt")
        print("✅ All agents saved.")

    return avg_mmr_history, max_mmr_history, env



def plot_behavior(agents):
    for agent in agents:
        plt.plot(agent.gold_spent_history, label=f"{agent.name} Gold Spent")
    plt.xlabel("Game")
    plt.ylabel("Gold Spent")
    plt.title("Gold Spent Over Games")
    plt.legend()
    plt.show()

    for agent in agents:
        plt.plot(agent.minions_bought_history, label=f"{agent.name} Minions Bought")
    plt.xlabel("Game")
    plt.ylabel("Minions Bought")
    plt.title("Minions Bought Over Games")
    plt.legend()
    plt.show()

    for agent in agents:
        plt.plot(agent.turns_skipped_history, label=f"{agent.name} Early End Turns")
    plt.xlabel("Game")
    plt.ylabel("Early End Turns")
    plt.title("Early End Turns Over Games")
    plt.legend()
    plt.show()


def print_behavior(agents):
    print("\n=== Agent Behavior Summary ===")
    for agent in agents:
        counts = agent.behavior_counts
        total = sum(counts.values())

        if total == 0:
            print(f"{agent.name}: No actions recorded yet.")
            continue

        print(f"\n{agent.name}:")
        for action, count in counts.items():
            percentage = (count / total) * 100
            print(f"  {action}: {count} times ({percentage:.1f}%)")


def plot_action_probabilities(agent, state):
    probs = agent.visualize_policy(state)  # Get action probabilities
    actions = ["buy_0", "buy_1", "buy_2", "roll", "level"]
    
    plt.plot(actions, probs.detach().numpy(), label=agent.name)
    plt.xlabel("Actions")
    plt.ylabel("Action Probabilities")
    plt.title(f"Policy Probabilities for {agent.name}")
    plt.legend()
    plt.show()





def plot_mmr_history(avg_mmr_history, max_mmr_history):
    plt.plot(avg_mmr_history, label='Average MMR')
    plt.plot(max_mmr_history, label='Highest MMR', linestyle='--')
    plt.xlabel('Batch')
    plt.ylabel('MMR')
    plt.title('Agent Learning Curve')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    avg_mmr_history, max_mmr_history, env = batch_train(num_batches=50, games_per_batch=20, print_every=5)
    plot_mmr_history(avg_mmr_history, max_mmr_history)

    # plot_behavior(env.agents)
