class Unit:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

class MultiplyGate:
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.top = Unit(x.value * y.value, 0)

        return self.top

    def backward(self):
        self.x.grad += self.y.value * self.top.grad
        self.y.grad += self.x.value * self.top.grad

class AddGate:
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.top = Unit(x.value + y.value, 0)

        return self.top

    def backward(self):
        self.x.grad += 1 * self.top.grad
        self.y.grad += 1 * self.top.grad

# ax + by + c
class Circuit:
    def __init__(self):
        self.multiply1 = MultiplyGate()
        self.multiply2 = MultiplyGate()

        self.add1 = AddGate()
        self.add2 = AddGate()

    def forward(self, x, y, a, b, c):
        self.ax = self.multiply1.forward(a, x)
        self.by = self.multiply2.forward(b, y)

        self.axpby = self.add1.forward(self.ax, self.by)
        self.axpbypc = self.add2.forward(self.axpby, c)

        return self.axpbypc

    def backward(self, topGrad):
        self.axpbypc.grad = topGrad;

        self.add2.backward()
        self.add1.backward()
        self.multiply2.backward()
        self.multiply1.backward()

class SVM:
    def __init__(self):
        self.circuit = Circuit()

        self.a = Unit(-1, 0)
        self.b = Unit(1, 0)
        self.c = Unit(2, 0)

    def forward(self, x, y):
        self.out = self.circuit.forward(Unit(x, 0), Unit(y, 0), self.a, self.b, self.c)
        return self.out

    def backward(self, label):
        self.a.grad = 0
        self.b.grad = 0
        self.c.grad = 0

        pull = 0

        if label == 1 and self.out.value < 1:
            pull = 1
        elif label == -1 and self.out.value > -1:
            pull = -1

        self.circuit.backward(pull)

        self.a.grad -= self.a.value
        self.b.grad -= self.b.value

    def learn_from(self, x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.update_parameters()

    def update_parameters(self):
        step = 0.001

        self.a.value += self.a.grad * step
        self.b.value += self.b.grad * step
        self.c.value += self.c.grad * step

svm = SVM()

svm.learn_from(1.2, 0.7, 1)
svm.learn_from(-0.3, -0.5, -1)
svm.learn_from(3.0, 0.1, 1)
svm.learn_from(-0.1, -1.0, -1)
svm.learn_from(-1.0, 1.1, -1)
svm.learn_from(2.1, -3, 1)
