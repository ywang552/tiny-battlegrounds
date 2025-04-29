# === simulate_game.py ===
import torch
from tiny_battlegrounds import TinyBattlegroundsEnv
from tiny_pbt import SelfLearningAgent  # Import your agent class

def simulate_single_game(agent):
    env = TinyBattlegroundsEnv([agent])

    env.setup()
    print("\n=== Starting Single Game Simulation ===\n")

    while sum(1 for a in env.agents if a.alive) > 0 and env.turn <= 20:
        for a in env.agents:
            if not a.alive:
                continue

            print(f"--- Turn {env.turn} ---")
            print(f"Gold: {a.gold}, Health: {a.health}, Tier: {a.tier}")

            if a.board:
                print("Board:")
                for m in a.board:
                    print(f"  {m.name} ({m.attack}/{m.health})")
            else:
                print("Board: EMPTY")

            if a.shop:
                print("Shop:")
                for idx, m in enumerate(a.shop):
                    print(f"  [{idx}] {m.name} ({m.attack}/{m.health})")
            else:
                print("Shop: EMPTY")

            base_gold = min(3 + (env.turn - 1), 10)  # 3 on Turn 1, 4 on Turn 2, etc, capped at 10
            agent.gold = base_gold

            a.shop = env.roll_shop(a.tier)

        while a.gold > 0:
            state = env.build_state_for_agent(a)
            available_actions = env.get_available_actions(a)

            if not available_actions:
                break

            action_idx = a.act(state)
            action_str = available_actions[action_idx % len(available_actions)]

            print(f"Action chosen: {action_str}")

            if action_str == "end_turn":
                print(">> Ending turn early.")
                break

            if action_str == "level":
                cost = env.get_upgrade_cost(a.tier)
                if a.gold >= cost:
                    a.gold -= cost
                    a.tier = min(a.tier + 1, 3)
                    print(f">> Upgraded to Tier {a.tier}.")
                continue

            if action_str.startswith("buy_"):
                idx = int(action_str.split("_")[1])
                if idx < len(a.shop) and a.gold >= 3 and len(a.board) < 7:
                    bought_minion = a.shop[idx]
                    a.gold -= 3
                    a.board.append(a.shop.pop(idx))
                    print(f">> Bought {bought_minion.name} ({bought_minion.attack}/{bought_minion.health})")
                continue

            if action_str == "roll":
                if a.gold >= 1:
                    a.gold -= 1
                    a.shop = env.roll_shop(a.tier)
                    print(f">> Rolled new shop.")
                continue

            if action_str.startswith("sell_"):
                idx = int(action_str.split("_")[1])
                if idx < len(a.board):
                    sold_minion = a.board.pop(idx)
                    a.gold += 1
                    print(f">> Sold {sold_minion.name} ({sold_minion.attack}/{sold_minion.health})")
                continue

            if agent.gold <= 0:
                break


        print()  # End of turn


        env.turn += 1

    print("\n=== Game Over ===")
    for a in env.agents:
        print(f"{a.name} Final Health: {a.health}")
        print("Final Board:")
        for m in a.board:
            print(f"  {m.name} ({m.attack}/{m.health})")


if __name__ == "__main__":
    # ✨ Load your trained agent
    agent = SelfLearningAgent(9, 5, name="MyTestAgent")
    agent.load("saved_models/agent_Bot_2_clone_clone_clone_clone_clone_clone_clone_clone_clone_final_20250429_032549.pt")  # Example path

    simulate_single_game(agent)
