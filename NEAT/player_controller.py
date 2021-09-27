# Simplified version of demo_controller that simply uses the output of a NEAT
# neural network structure to calculate the necessary actions

import sys
sys.path.insert(0, 'evoman')

from controller import Controller
import numpy as np

# implements controller structure for player
class player_controller(Controller):

	def control(self, inputs, controller):
		# Normalise the input using min-max scaling:
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		output = controller.activate(inputs) # controller is a neural network of type neat.nn.FeedForwardNetwork

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]
