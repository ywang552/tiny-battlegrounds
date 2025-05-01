import random
import torch
import matplotlib.pyplot as plt  # Add this at the top of your file if not imported yet
import os
import json
from utils.load import load_minions
from agents.transformer_agent import TransformerAgent



TRIBE_TO_INDEX = {
    "Beast": 0,
    "Mech": 1,
    "Murloc": 2,
    "Demon": 3,
    "Dragon": 4,
    "Elemental": 5,
    "Naga": 6,
    "Quilboar": 7,
    "Pirate": 8,
    "Undead": 9,
}


def encode_minion(minion, source_flag, slot_idx=None):
    tribes = torch.zeros(10)  # Assuming 10 tribe slots (no 'None')

    if minion is not None:
        types = minion.types or []

        if "All" in types:
            tribes[:] = 1
        else:
            for tribe in types:
                if tribe is None:
                    continue  # safely skip [None]
                if tribe in TRIBE_TO_INDEX:
                    tribes[TRIBE_TO_INDEX[tribe]] = 1


        base = torch.tensor([
            float(minion.attack),
            float(minion.health),
            float(minion.tier),
        ], dtype=torch.float32)

        slot_feature = torch.tensor(
            [slot_idx / 7.0], dtype=torch.float32
        ) if slot_idx is not None else torch.tensor([0.0])

        out = torch.cat([base, tribes, torch.tensor([source_flag], dtype=torch.float32), slot_feature])
        return out

    except Exception as e:
        print(f"❌ Failed to encode minion: {minion.name}, error: {e}")
        return torch.zeros(16)



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
            agent.gold_spent_this_game = 0
            agent.minions_bought_this_game = 0
            agent.turns_skipped_this_game = 0
            agent.behavior_counts = {'buy': 0, 'sell': 0, 'roll': 0, 'level': 0, 'end_turn': 0}
            agent.tavern_upgrade_cost = self.get_base_upgrade_cost(agent.tier)  # ✅ One-time init
            if isinstance(agent, TransformerAgent):
                agent.env = self

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
            assert self.latest_dead_agent is not None, "Ghost pairing requested before any agent has died"
            ghost_opponent = self.latest_dead_agent
            pairs.append((ghost_fighter, ghost_opponent))


        for i in range(0, len(alive_agents), 2):
            pairs.append((alive_agents[i], alive_agents[i+1]))

        return pairs

    def simulate_combat(self, a, d):
        m1 = sum(m.strength() for m in a.board)
        m2 = sum(m.strength() for m in d.board)

        # Tie case
        if m1 == m2:
            if a.alive:
                state = a.build_state(self, phase="combat", previous_health=a.health, enemy_strength=m2)
                a.observe(state, 0.0)
            if d.alive:
                state = d.build_state(self, phase="combat", previous_health=d.health, enemy_strength=m1)
                d.observe(state, 0.0)
            return

        # Determine winner and loser
        winner = a if m1 > m2 else d
        loser  = d if m1 > m2 else a

        # Winner reward
        if winner.alive:
            state = winner.build_state(self, phase="combat", previous_health=winner.health, enemy_strength=sum(m.strength() for m in loser.board))
            winner.observe(state, 5.0, opponent=loser)  # ✅ PASS OPPONENT

        # Loser punishment
        if loser.alive:
            damage = winner.tier + sum(m.tier for m in winner.board)
            loser.health -= max(damage, 1)
            state = loser.build_state(self, phase="combat", previous_health=loser.health + max(damage, 1), enemy_strength=sum(m.strength() for m in winner.board))
            loser.observe(state, -5.0, opponent=winner)  # ✅ PASS OPPONENT



    def remove_dead(self):
        latest_turn = self.turn
        highest_health_before_death = float('-inf')
        best_candidate = None  # ✅ FIXED

        for agent in self.agents:
            if agent.alive and agent.health <= 0:
                prev_health = agent.health  # already <= 0
                agent.health = max(agent.health, 0)
                agent.alive = False
                self.dead[agent] = latest_turn

                # ✅ Tie-breaker logic:
                if prev_health > highest_health_before_death:
                    best_candidate = agent
                    highest_health_before_death = prev_health

        # ✅ Store best candidate as ghost agent
        if best_candidate:
            self.latest_dead_agent = best_candidate

    def build_active_state(self, agent):
        gold = agent.gold
        gold_cap = agent.gold_cap
        tier = agent.tier
        health = agent.health
        turn_number = self.turn

        board_strength = sum(m.strength() for m in agent.board)
        num_minions = len(agent.board)

        shop_slots = [m.tier for m in agent.shop]
        while len(shop_slots) < 6:
            shop_slots.append(0)

        # Pad unused combat-related features
        health_delta = 0.0
        previous_health = 0.0
        enemy_strength = 0.0

        state_vector = [
            gold, gold_cap, tier, health, turn_number,
            board_strength, num_minions,
            *shop_slots,              # 6 elements
            previous_health,          # combat-only
            health_delta,             # combat-only
            enemy_strength            # combat-only
        ]

        # Pad up to 20 if we add more features later
        while len(state_vector) < 20:
            state_vector.append(0.0)

        return torch.tensor(state_vector, dtype=torch.float32)
    
    def build_combat_state(self, agent, previous_health, enemy_strength):
        current_health = agent.health
        health_delta = previous_health - current_health
        turn_number = self.turn
        tier = agent.tier
        board_strength = sum(m.strength() for m in agent.board)
        num_minions = len(agent.board)
        gold_cap = agent.gold_cap

        # Pad shop-related fields
        gold = 0.0
        shop_slots = [0.0] * 6  # zero for shop_slot_0 to shop_slot_5

        state_vector = [
            gold, gold_cap, tier, current_health, turn_number,
            board_strength, num_minions,
            *shop_slots,         # inactive phase doesn't access shop
            previous_health,
            health_delta,
            enemy_strength
        ]

        while len(state_vector) < 20:
            state_vector.append(0.0)

        return torch.tensor(state_vector, dtype=torch.float32)

    def build_transformer_state(self, agent, phase="active"):
        # === Basic info ===
        state_vec = torch.tensor([
            agent.tier,
            agent.health,
            self.turn
        ], dtype=torch.float32)

        # === Econ info ===
        if phase == "active":
            econ_vec = torch.tensor([
                agent.gold,
                agent.gold_cap,
                1.0,               # fixed roll cost
                agent.upgrade_cost
            ], dtype=torch.float32)

            shop_minions = [
                encode_minion(m, source_flag=0, slot_idx=3 + i)
                for i, m in enumerate(agent.shop)
            ]
        else:
            econ_vec = torch.tensor([0.0, agent.gold_cap, 0.0, 0.0], dtype=torch.float32)
            shop_minions = []

        # === Tier vector ===
        tier_vec = torch.tensor([
            1 if i < agent.tier else 0 for i in range(6)
        ], dtype=torch.float32)

        # === Board minions ===
        board_minions = [
            encode_minion(m, source_flag=1, slot_idx=i)
            for i, m in enumerate(agent.board)
        ]

        # === Opponent summary ===
        opponent = self.current_opponent.get(agent, None)
        opponent_vec = None
        if opponent and hasattr(agent, "opponent_memory") and opponent in agent.opponent_memory:
            opponent_vec = torch.tensor(agent.opponent_memory[opponent], dtype=torch.float32)

        # === NaN cleanup just in case ===
        econ_vec = torch.nan_to_num(econ_vec)

        # === Token assembly ===
        token_inputs = {
            "state_vec": state_vec,
            "econ_vec": econ_vec,
            "tier_vec": tier_vec,
            "board_minions": board_minions,
            "shop_minions": shop_minions,
            "opponent_vec": opponent_vec
        }

        return agent.build_tokens(**token_inputs)

    def build_state(self, agent, phase="active"):
        if isinstance(agent, TransformerAgent):
            player_id = self.agents.index(agent)
            return self.build_transformer_state(agent, phase)
        else:
            return self.build_active_state(agent) if phase == "active" else self.build_combat_state(agent)


    def step(self, verbose=False, focus_agent_name=None):
        # ⚔️ --- Match Making ---
        alive_agents = [a for a in self.agents if a.alive]
        pairs = self.matchmaking(alive_agents)
        self.current_opponent = {}

        for p1, p2 in pairs:
            if p2 is None:
                self.current_opponent[p1] = None

            else:
                self.current_opponent[p1] = p2
                self.current_opponent[p2] = p1

        # === ACTIVE PHASE ===
        for agent in self.agents:
            if not agent.alive:
                continue

            agent.shop = self.roll_shop(agent.tier)
            agent.gold_cap = min(3 + self.turn - 1, 10)
            agent.gold = agent.gold_cap



            if verbose and agent.name == focus_agent_name:
                print(f"\n--- Turn {self.turn} ---")
                print(f"Gold: {agent.gold}, Health: {agent.health}, Tier: {agent.tier}")
                print("Board:" if agent.board else "Board: EMPTY")
                for m in agent.board:
                    print(f"  {m.name} ({m.attack}/{m.health})")
                print(f"Shop of length {len(agent.shop)}:" if agent.shop else "Shop: EMPTY")
                for idx, m in enumerate(agent.shop):
                    print(f"  [{idx}] {m.name} ({m.attack}/{m.health})")

            while agent.gold > 0:
                available_actions = self.get_available_actions(agent)

                if not available_actions:
                    break
                
                self.build_state()
                action = agent.act(state)
                action_str = available_actions[action % len(available_actions)]

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

                    agent.observe(agent.build_state(self, phase="active"), 0.0)
                    break

                if action_str == "level":
                    cost = agent.tavern_upgrade_cost
                    if agent.gold >= cost:
                        agent.gold -= cost
                        agent.tier += 1
                        agent.tavern_upgrade_cost = self.get_base_upgrade_cost(agent.tier)
                        agent.gold_spent_this_game += cost
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Upgraded to Tier {agent.tier}.")
                    agent.observe(agent.build_state(self, phase="active"), 0.0)
                    continue

                if action_str.startswith("buy_"):
                    idx = int(action_str.split("_")[1])
                    if idx < len(agent.shop) and agent.gold >= 3 and len(agent.board) < 7:
                        bought_minion = agent.shop[idx]
                        agent.gold -= 3
                        agent.board.append(agent.shop.pop(idx))
                        agent.minions_bought_this_game += 1
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Bought {bought_minion.name} ({bought_minion.attack}/{bought_minion.health})")
                    agent.observe(agent.build_state(self, phase="active"), 0.0)
                    continue

                if action_str == "roll":
                    if agent.gold >= 1:
                        agent.gold -= 1
                        agent.shop = self.roll_shop(agent.tier)
                        agent.gold_spent_this_game += 1
                    agent.observe(agent.build_state(self, phase="active"), 0.0)
                    continue

                if action_str.startswith("sell_"):
                    idx = int(action_str.split("_")[1])
                    if idx < len(agent.board):
                        sold_minion = agent.board.pop(idx)
                        agent.gold += 1
                        if verbose and agent.name == focus_agent_name:
                            print(f">> Sold {sold_minion.name} ({sold_minion.attack}/{sold_minion.health})")
                    agent.observe(agent.build_state(self, phase="active"), 0.0)
                    continue

        # Reduce upgrade cost
        for agent in self.agents:
            old_cost = agent.tavern_upgrade_cost
            agent.tavern_upgrade_cost = max(agent.tavern_upgrade_cost - 1, 0)
            if verbose and agent.name == focus_agent_name:
                print(f"[Turn {self.turn}] {agent.name} (Tier {agent.tier}) upgrade cost: {old_cost} → {agent.tavern_upgrade_cost}")

        # ⚔️ --- COMBAT PHASE ---
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
