import NeuralNetwork from './NeuralNetwork';
import NeuronLayer from './NeuronLayer';
import { ActivateFunction, ActiveFunctions } from './ActivateFunctions';

const hiddenLayers = [
    new NeuronLayer(2, ActiveFunctions[ActivateFunction.Sigmoid], 0.35),
    new NeuronLayer(3, ActiveFunctions[ActivateFunction.Sigmoid], 0.8),
];

const outputLayer = new NeuronLayer(2, ActiveFunctions[ActivateFunction.Sigmoid], 0.6);

const neuralNetwork = new NeuralNetwork(2, hiddenLayers, outputLayer);

for (let i = 0; i < 10000; i++) {
    neuralNetwork.train([0.05, 0.1], [0.01, 0.99]);
    console.log(i, neuralNetwork.calculateTotalError([[[0.05, 0.1], [0.01, 0.99]]]));
}
