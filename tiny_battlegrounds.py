import random
import torch
from models.tiny_nn_policy import SelfLearningAgent  # <-- import your self-learning agent!
import matplotlib.pyplot as plt  # Add this at the top of your file if not imported yet

# --- Minion Class ---
class Minion:
    def __init__(self, name, tribe, attack, health, tier):
        self.name = name
        self.tribe = tribe
        self.attack = attack
        self.health = health
        self.tier = tier

    def strength(self):
        return self.attack + self.health

    def __repr__(self):
        return f"{self.name}({self.attack}/{self.health})"

# --- Minion Pool (Tiny version) ---
MINION_POOL = [
    Minion("Microbot", "Mech", 1, 1, 1),
    Minion("Kaboom Bot", "Mech", 2, 2, 2),
    Minion("Iron Sensei", "Mech", 2, 3, 2),
    Minion("Deflect-o-Bot", "Mech", 3, 2, 3),
    Minion("Sneed's Assistant", "Mech", 4, 3, 3),
    Minion("Red Whelp", "Dragon", 2, 1, 1),
    Minion("Glyph Guardian", "Dragon", 2, 4, 2),
    Minion("Bronze Warden", "Dragon", 2, 1, 2),
    Minion("Herald of Flame", "Dragon", 3, 5, 3),
    Minion("Drakonid Enforcer", "Dragon", 3, 6, 3),
]

# --- TinyBattlegrounds Environment ---
class TinyBattlegroundsEnv:
    def __init__(self, agents):
        self.agents = agents
        self.turn = 1
        self.dead = {}
        self.ghost_board = None

    def setup(self):
        for agent in self.agents:
            agent.gold = 3
            agent.tier = 1
            agent.board = []
            agent.health = 40
            agent.alive = True
            agent.shop = self.roll_shop(agent.tier)

    def roll_shop(self, tier):
        pool = [m for m in MINION_POOL if m.tier <= tier]
        return random.sample(pool, min(3, len(pool)))

    def get_upgrade_cost(self, current_tier):
        if current_tier == 1:
            return 5
        elif current_tier == 2:
            return 7
        else:
            return 999

    def get_available_actions(self, agent):
        actions = []
        upgrade_cost = self.get_upgrade_cost(agent.tier)

        if agent.gold >= upgrade_cost:
            actions.append("level")

        for idx, minion in enumerate(agent.shop):
            if agent.gold >= 3 and len(agent.board) < 7:
                actions.append(f"buy_{idx}")

        if agent.gold >= 1:
            actions.append("roll")

        return actions

    def matchmaking(self, alive_agents):
        pairs = []
        random.shuffle(alive_agents)
        if len(alive_agents) % 2 == 1:
            ghost_fighter = alive_agents.pop()
            pairs.append((ghost_fighter, 'ghost'))

        for i in range(0, len(alive_agents), 2):
            pairs.append((alive_agents[i], alive_agents[i + 1]))

        return pairs

    def simulate_combat(self, attacker, defender):
        attacker_strength = sum(m.strength() for m in attacker.board)
        if defender == 'ghost':
            defender_strength = sum(m.strength() for m in self.ghost_board) if self.ghost_board else 0
        else:
            defender_strength = sum(m.strength() for m in defender.board)

        if attacker_strength > defender_strength:
            attacker.cumulative_reward += 2.0  # Reward for winning combat
            if defender != 'ghost':
                defender.cumulative_reward -= 1.0  # Penalty for losing combat

        elif attacker_strength < defender_strength:
            attacker.cumulative_reward -= 1.0  # Penalty for losing combat
            if defender != 'ghost':
                defender.cumulative_reward += 2.0  # Reward for winning combat

        else:
            attacker.cumulative_reward += 0.5  # Small reward for tying
            if defender != 'ghost':
                defender.cumulative_reward += 0.5  # Small reward for tying



    def remove_dead(self):
        for agent in self.agents:
            if agent.alive and agent.health <= 0:
                agent.alive = False
                self.dead[agent] = self.turn
                if not self.ghost_board or random.random() < 0.5:
                    self.ghost_board = list(agent.board)

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



    def step(self):
        for agent in self.agents:
            if not agent.alive:
                continue

            # Shop Phase
            agent.gold = min(agent.gold + 1, 10)
            agent.shop = self.roll_shop(agent.tier)

            while agent.gold > 0:
                state = self.build_state_for_agent(agent)
                action_idx = agent.act(state)  # Choose action
                available_actions = self.get_available_actions(agent)

                if not available_actions:
                    break
                action_str = available_actions[action_idx % len(available_actions)]

                if action_str == "level":
                    cost = self.get_upgrade_cost(agent.tier)
                    if agent.gold >= cost:
                        agent.gold -= cost
                        agent.tier = min(agent.tier + 1, 3)
                        agent.cumulative_reward += 2.0  # Reward for leveling

                elif action_str.startswith("buy_"):
                    idx = int(action_str.split("_")[1])
                    if idx < len(agent.shop) and agent.gold >= 3 and len(agent.board) < 7:
                        agent.gold -= 3
                        agent.board.append(agent.shop.pop(idx))
                        agent.cumulative_reward += 1.0  # Reward for buying a minion

                elif action_str == "roll":
                    if agent.gold >= 1:
                        agent.gold -= 1
                        agent.shop = self.roll_shop(agent.tier)
                        # No reward for rolling, can add if needed

            # Combat Phase
            alive_agents = [a for a in self.agents if a.alive]
            pairs = self.matchmaking(alive_agents)

            for p1, p2 in pairs:
                if p1.alive:
                    self.simulate_combat(p1, p2)
                if p2 != 'ghost' and p2.alive:
                    self.simulate_combat(p2, p1)

            self.remove_dead()



    def play_game(self):
        self.setup()
        while sum(1 for agent in self.agents if agent.alive) > 1 and self.turn <= 50:
            self.step()
            self.turn += 1

        for agent in self.agents:
            if agent.alive and agent not in self.dead:
                self.dead[agent] = self.turn

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
            current_place -= tied_size

        return rewards


def batch_train(num_batches=100, games_per_batch=10, print_every=10):
    agents = [SelfLearningAgent(input_size=9, action_size=5, name=f"Bot_{i}") for i in range(8)]
    mmr_history = []

    # Create the environment here
    env = TinyBattlegroundsEnv(agents)

    for batch in range(num_batches):
        total_rewards = {agent.name: 0 for agent in agents}

        for game in range(games_per_batch):
            env.setup()  # Reset game state for each batch
            rewards = env.play_game()

            for agent in agents:
                agent.learn(rewards[agent.name])
                total_rewards[agent.name] += rewards[agent.name]

        avg_mmr = sum(total_rewards.values()) / len(agents)
        mmr_history.append(avg_mmr)

        if (batch + 1) % print_every == 0:
            print(f"Batch {batch + 1}: Average MMR = {avg_mmr:.2f}")
            for agent in agents:
                state = env.build_state_for_agent(agent)  # Use the env instance here
                plot_action_probabilities(agent, state)

    return mmr_history


def plot_action_probabilities(agent, state):
    probs = agent.visualize_policy(state)  # Get action probabilities
    actions = ["buy_0", "buy_1", "buy_2", "roll", "level"]
    
    plt.plot(actions, probs.detach().numpy(), label=agent.name)
    plt.xlabel("Actions")
    plt.ylabel("Action Probabilities")
    plt.title(f"Policy Probabilities for {agent.name}")
    plt.legend()
    plt.show()





def plot_mmr_history(mmr_history):
    plt.plot(mmr_history)
    plt.xlabel('Batch')
    plt.ylabel('Average MMR')
    plt.title('Agent Learning Curve')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    mmr_history = batch_train(num_batches=100, games_per_batch=10, print_every=10)
    plot_mmr_history(mmr_history)
