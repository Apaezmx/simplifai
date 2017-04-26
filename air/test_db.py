import unittest
from db import *
from config import config
from model import *

class TestDB(unittest.TestCase):

    def test_model(self):
      model = new_model(config.ROOT_PATH)
      self.assertTrue(model)
      self.assertEquals(model.status, ModelStatus.CREATED)
      model.status = ModelStatus.NULL
      save_model(model)
      model = get_model(config.ROOT_PATH, model.get_handle())
      self.assertEqual(model.status, ModelStatus.NULL)


if __name__ == '__main__':
    unittest.main()
