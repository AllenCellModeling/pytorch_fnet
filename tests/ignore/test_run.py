import unittest
import subprocess

class TestRun(unittest.TestCase):
    def test_test_run_script(self):
        """Test running test_run.sh."""
        subprocess.run(['./scripts/test_run.sh', '0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
