# Sentence to Quantum Circuit Generator

from lambeq import BobcatParser, SpacyTokeniser, AtomicType, IQPAnsatz
from discopy import grammar
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.qiskit import tk_to_qiskit
import matplotlib.pyplot as plt

print("\n")
print("#################################################")
print("### Sentence to Quantum Circuit Transformator ###")
print("#################################################")

# Sentence preparation

# Example text
example_text = "I love pizza. It is my favorite food."
print("\nText: ", example_text)

tokeniser = SpacyTokeniser()
sentences = tokeniser.split_sentences(example_text)
print("\nSentences: ", sentences)

sentence1 = sentences[0]
print("\nSentence 1: ", sentence1)

# Get user input
sentence2 = input("\nType in the sentence to be transformed into a quantum circuit:...")

# Sentence diagram

parser = BobcatParser(verbose='suppress')
diagram = parser.sentence2diagram(sentence2, tokenised=False)

answer1 = input("\nPress 'y' to show the sentence diagram for your sentence...")
if answer1 in ['y', 'Y']: 
    grammar.draw(diagram, figsize=(23,4), fontsize=12)

# Quantum circuit

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Convert string diagram to quantum circuit
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=2)
discopy_circuit = ansatz(diagram)

answer2 = input("\nPress 'y' to show the raw quantum circuit diagram for your sentence. The final quantum circuit diagram will be saved as file 'q-circuit.jpg'")
if answer2 in ['y', 'Y']:
    discopy_circuit.draw(figsize=(15,10))

# Conversion of the diagram to pytket format & export to obtain a Qiskit circuit
tket_circuit = discopy_circuit.to_tk()
qiskit_circuit = tk_to_qiskit(tket_circuit)
plt = qiskit_circuit.draw(output="mpl")
plt.show()
plt.savefig('q-circuit.jpg')
