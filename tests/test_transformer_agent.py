import torch
import unittest
from types import SimpleNamespace

from env.tiny_battlegrounds import TinyBattlegroundsEnv
from agents.transformer_agent import TransformerAgent

class TestTransformerAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize a mock environment and transformer agent
        cls.env = TinyBattlegroundsEnv([])
        cls.agent = TransformerAgent()

        # Create a mock minion
        cls.mock_minion = SimpleNamespace(
            attack=3,
            health=2,
            tier=1,
            types=["Beast"]
        )

    def test_encode_minion_basic(self):
        """Test encoding of a standard minion with known tribe."""
        encoded = TinyBattlegroundsEnv.encode_minion(
            self.mock_minion,
            source_flag=1,
            slot_idx=2
        )
        self.assertEqual(encoded.shape, (16,))
        self.assertEqual(encoded[0].item(), 3.0)  # Attack
        self.assertEqual(encoded[3].item(), 1.0)  # "Beast" tribe is index 0 + 3 = 3

    def test_encode_minion_none_type(self):
        """Test encoding when tribe is None."""
        dummy = SimpleNamespace(attack=0, health=0, tier=1, types=[None])
        encoded = TinyBattlegroundsEnv.encode_minion(dummy, 0, None)
        self.assertEqual(encoded.shape, (16,))
        self.assertEqual(encoded[13].item(), 1.0)  # Tribe "None" is index 10 + 3 = 13

    def test_active_phase_state_tokens(self):
        """Test transformer token building in active phase using a real agent instance."""

        # Use real agent to ensure hashability
        test_agent = TransformerAgent(name="TestAgent")
        test_agent.health = 25
        test_agent.tier = 2
        test_agent.gold = 3
        test_agent.gold_cap = 10
        test_agent.board = [self.mock_minion]
        test_agent.shop = [self.mock_minion]
        test_agent.opponent_memory = {}
        test_agent.tavern_upgrade_cost = 5

        opponent = TransformerAgent(name="OpponentAgent")
        opponent.tier = 3
        opponent.alive = True

        self.env.current_opponent = {test_agent: opponent}

        state = self.env.build_transformer_state(test_agent, "active")

        # Validate structure of returned token dict
        self.assertIn("tokens", state)
        self.assertIn("masks", state)
        self.assertEqual(state["tokens"]["state"].shape, (3,))
        self.assertEqual(state["tokens"]["board"].shape, (7, 16))
        self.assertTrue(state["masks"]["action"].any())



if __name__ == "__main__":
    unittest.main()
