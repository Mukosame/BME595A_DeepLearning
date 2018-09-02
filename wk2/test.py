from neural_network import NeuralNetwork
from logic_gates import AND, OR, NOT, XOR

And = AND()
Not = NOT()
Or = OR()
Xor = XOR()

print(And(True, True))
print(And(True, False))
print(And(False, False))
print(Not(False))
print(Not(True)) 
print(Or(True, True))
print(Or(True, False))
print(Or(False, False))
print(Xor(True, True))
print(Xor(True, False))
print(Xor(False, False))