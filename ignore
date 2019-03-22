import pytest
import subprocess
import torch


def test_train_cpu():
    subprocess.run(
        ['./scripts/test_run.sh', '-1'],
        check=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='gpu required')
def test_train_gpu():
    subprocess.run(
        ['./scripts/test_run.sh', '0'],
        check=True,
    )
