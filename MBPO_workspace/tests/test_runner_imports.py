import unittest
from pathlib import Path

class ArchitectureTest(unittest.TestCase):
    def test_modern_pipeline_no_ray_or_tf(self):
        root=Path(__file__).resolve().parents[1]/'src'
        for name in ('config.py','trainer.py','dynamics.py','replay.py','evaluation.py','plots.py'):
            text=(root/name).read_text(encoding='utf-8')
            self.assertNotIn('import tensorflow',text)
            self.assertNotIn('from ray import',text)
            self.assertNotIn('import ray',text)
    def test_shared_sac_environment_and_optimizer(self):
        root=Path(__file__).resolve().parents[1]/'src'
        source=(root/'trainer.py').read_text()
        self.assertIn('from SAC_workspace.src.agents.sac_agent import SACAgent',source)
        self.assertIn('from SAC_workspace.src.environment.environment import SACEnvironment',source)
        self.assertIn('mix_batches',source)

if __name__=='__main__':unittest.main()
