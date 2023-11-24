import numpy as np


class FixedStepIntegrator:
	def __init__(self, rhs, step, t0, x0):
		self.rhs = rhs
		assert step > 0
		self.step = step
		self.t = float(t0)
		self.y = np.copy(x0)

	def successful(self):
		return np.all(np.isfinite(self.y))

	def __process_forward(self, t):
		while self.t + self.step <= t:
			self.y += self.rhs(self.t, self.y) * self.step
			self.t += self.step

		if self.t < t:
			self.y += self.rhs(self.t, self.y) * (t - self.t)
			self.t = t

		return self.y

	def __process_backward(self, t):
		while self.t - self.step >= t:
			self.y -= self.rhs(self.t, self.y) * self.step
			self.t += self.step

		if self.t > t:
			self.y += self.rhs(self.t, self.y) * (t - self.t)
			self.t = t

		return self.y

	def integrate(self, t):
		if t >= self.t:
			return self.__process_forward(t)
		return self.__process_backward(t)

	def __call__(self, t):
		return self.integrate(t)
