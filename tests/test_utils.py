from fnet.utils.general_utils import str_to_object
import random
import pdb


def _dummy():
    print('Hi')


def test_str_to_object():
    exp = [_dummy, random.randrange]
    for idx_s, as_str in enumerate(['_dummy', 'random.randrange']):
        obj = str_to_object(as_str)
        assert obj is exp[idx_s], f'{obj} is not {exp[idx_s]}'


if __name__ == '__main__':
    test_str_to_object()
