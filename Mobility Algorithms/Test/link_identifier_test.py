import unittest
import numpy as np
from bsm_stream_vector import LinkIdentifier

"""Test LinkIndentifier class in bsm_stream_vector to ensure the BSMs are being assigned to the correct Measures Estimation link 
"""

class LinkIdentifierTest(unittest.TestCase):
	#This test only works if findLink in LinkIndentifier returns [0,7]
	def setUp(self):
		self.link_identifier = LinkIdentifier("i405links.csv")

	def test_findLink(self):
		"""Test that BSMs in bsm_sample.csv are assigned to the correct Measures Estimation link
		"""
		with open("bsm_sample.csv") as in_f:
			is_header = True
			for row in in_f:
				if is_header:
					is_header = False
					continue
				data = row.split(',')
				point = np.array([float(data[14]),float(data[15])])
				link = int(data[8])
				self.assertTrue(link == self.link_identifier.findLink(point))


if __name__ == '__main__':
    unittest.main()