import unittest
from db import *
from config import config
from model import *
from mockcache import Client

class TestDB(unittest.TestCase):
    def setUp(self):
      config.mc = Client()

    def test_model(self):
      model = new_model()
      self.assertTrue(model)
      self.assertEquals(model.status, ModelStatus.CREATED)
      model.status = ModelStatus.NULL
      save_model(model)
      model = get_model(model.get_handle())
      self.assertEqual(model.status, ModelStatus.NULL)
      self.assertEqual(delete_model(model.get_handle()), 1)


if __name__ == '__main__':
    unittest.main()
