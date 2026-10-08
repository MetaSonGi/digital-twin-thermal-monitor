import importlib.util, json, threading, unittest, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("app",ROOT/"main.py")
thermal=importlib.util.module_from_spec(spec)
spec.loader.exec_module(thermal)

class ThermalTests(unittest.TestCase):
    def test_held_out_statuses(self):
        r=thermal.run(ROOT/'sample.csv')
        self.assertEqual([x['status'] for x in r['results']],['normal','anomaly','out_of_domain'])
    def test_degenerate_training(self):
        with self.assertRaises(ValueError): thermal.fit([{'load':1,'temperature':30,'ambient':25}]*4)
    def test_invalid_split(self):
        with self.assertRaises(ValueError): thermal.run(ROOT/'sample.csv',15)

if __name__=="__main__": unittest.main()
