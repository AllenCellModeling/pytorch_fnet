import unittest
import util.data

class TestDataSetCellnucNotCellnuc(unittest.TestCase):

    def test_simple(self):
        path = 'data/few_files'
        dataset = util.data.DataSetCellnucNotCellnuc(path, train=True)
        print(dataset)
        print()
        data = dataset[0]
        self.assertEqual(len(data), 4)
