from neural_network import NeuralNetwork
from logic_gates import AND, OR, NOT, XOR

And = AND()
Not = NOT()
Or = OR()
Xor = XOR()

And.train()

print(And(True, True)) #True
print(And(True, False)) #False
print(And(False, False)) #False
'''
print(Not(False))#True
print(Not(True)) #False
print(Or(True, True))#True
print(Or(True, False))#True
print(Or(False, False))#False
print(Xor(True, True))#False
print(Xor(True, False))#True
print(Xor(False, False))#False
'''