import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.tiny_battlegrounds import TinyBattlegroundsEnv


class MockAgent:
    def __init__(self, name):
        self.name = name
        self.board = []
        self.shop = []
        self.gold = 10
        self.health = 40
        self.memory = []

    def act(self, *args, **kwargs):
        # Always end turn (16 is "end turn" action in your env)
        return 16, torch.tensor(0.0), torch.tensor(0.0)

    def observe(self, *args, **kwargs):
        pass

    def reset(self):
        self.board = []
        self.shop = []
        self.gold = 10
        self.health = 40
        self.memory = []


def test_env_initialization():
    agents = [MockAgent(name=f"Agent_{i}") for i in range(8)]
    env = TinyBattlegroundsEnv(agents)
    assert len(env.agents) == 8
    for agent in env.agents:
        assert hasattr(agent, "board")
        assert hasattr(agent, "shop")

def test_decode_action_compatible_with_step_logic():
    agent = MockAgent("TestAgent")
    env = TinyBattlegroundsEnv([agent])


    assert env.decode_action(agent, env.BUY_START) == "buy_0"
    assert env.decode_action(agent, env.BUY_START + 5) == "buy_6"
    assert env.decode_action(agent, env.SELL_START) == "sell_0"
    assert env.decode_action(agent, env.SELL_START + 5 - 1) == "sell_5"
    assert env.decode_action(agent, env.LEVEL_IDX) == "level"
    assert env.decode_action(agent, env.ROLL_IDX) == "roll"
    assert env.decode_action(agent, env.END_TURN_IDX) == "end_turn"

    # Ensure invalid index raises error
    try:
        env.decode_action(agent, env.ACTION_SIZE)
        assert False, "Expected exception"
    except ValueError:
        pass



    
