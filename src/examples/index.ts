import NeuralNetwork from '../NeuralNetwork';
import NeuronLayer from '../NeuronLayer';
import { ActivateFunction, ActiveFunctions } from '../ActivateFunctions';
import { features, labels } from './trainExamples';

const imageSize = 28 * 28;
const classesNumber = 10; // ten unique numbers

const hiddenLayers = [
    new NeuronLayer(imageSize, ActiveFunctions[ActivateFunction.Sigmoid]),
    new NeuronLayer(32, ActiveFunctions[ActivateFunction.Sigmoid]),
];

const outputLayer = new NeuronLayer(classesNumber, ActiveFunctions[ActivateFunction.Softmax]);

const neuralNetwork = new NeuralNetwork(imageSize, hiddenLayers, outputLayer);

for (let i = 1; i <= 5; i++) {
    features.forEach((feature, i) => {
        neuralNetwork.train(feature, labels[i]);
    });

    console.log(
        i,
        features.reduce((sum, feature, i) => sum + neuralNetwork.calculateTotalError([[feature, labels[i]]]), 0) / features.length
    );
}

// neuralNetwork.train(features[1], labels[1]);
