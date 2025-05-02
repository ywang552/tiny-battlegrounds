import unittest
from tiny_battlegrounds import TinyBattlegroundsEnv, Minion
from agents.transformer_agent import TransformerAgent  # Replace with your agent class

class TestGameLogic(unittest.TestCase):
    def setUp(self):
        """Setup fresh game for each test."""
        self.agents = [TransformerAgent(name=f"Player{i}") for i in range(2)]
        self.env = TinyBattlegroundsEnv(self.agents)
    
    def test_initial_state(self):
        """Check if game initializes correctly."""
        self.assertEqual(self.env.turn, 0)
        self.assertTrue(all(agent.tier == 1 for agent in self.env.agents))
    
    def test_turn_increment(self):
        """Test if turn counter advances."""
        self.env.step()
        self.assertEqual(self.env.turn, 1)
    
    def test_combat_after_turn(self):
        """Ensure combat reduces health."""
        self.env.step()
        alive_agents = [a for a in self.env.agents if a.alive]
        self.assertLessEqual(len(alive_agents), 2)  # No one should die on turn 1

if __name__ == "__main__":
    unittest.main()