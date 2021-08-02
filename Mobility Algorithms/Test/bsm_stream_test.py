import unittest
from bsm_stream_vector import BSM, BSMStream

"""Test class for BSMStream confirming data falls within appropriate boundaries and links are assigned correctly
"""

class BSMStreamTest(unittest.TestCase):
	def setUp(self):
		self.bsm_stream = BSMStream("bsm_sample.csv","i405links_tracking.csv")

	def testRead(self):
		iteration = 0.2
		for tp, bsms in self.bsm_stream.read():
			for bsm in bsms:
				if bsm[BSM.RouteIndex] != 0:
					self.assertTrue(tp - 5 <= bsm[BSM.TimeIndex] <= tp)
					self.assertTrue(-91863 <= bsm[BSM.XIndex] <= 9842)
					self.assertTrue(-113189 <= bsm[BSM.YIndex] <= 149278)
					self.assertTrue(0 <= bsm[BSM.SpeedIndex] <= 70)
					self.assertTrue(bsm[BSM.LinkIndex] == bsm[BSM.VISSIMIndex])
			iteration += 0.2

if __name__ == '__main__':
    unittest.main()