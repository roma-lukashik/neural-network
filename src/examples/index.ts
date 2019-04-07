import NeuralNetwork from '../NeuralNetwork';
import NeuronLayer from '../NeuronLayer';
import { ActivateFunction, ActiveFunctions } from '../ActivateFunctions';
import { trainingData } from './trainExamples';

const imageSize = 28 * 28;
const classesNumber = 10; // ten unique numbers

const hiddenLayers = [
    new NeuronLayer(100, ActiveFunctions[ActivateFunction.Sigmoid]),
];

const outputLayer = new NeuronLayer(classesNumber, ActiveFunctions[ActivateFunction.Softmax]);

const neuralNetwork = new NeuralNetwork(imageSize, hiddenLayers, outputLayer);

const trainings = [...trainingData];

const epochs = 5;

for (let i = 1; i <= epochs; i++) {
    trainings.forEach(([feature, label]) => {
        for (let j = 0; j < 5; j++) {
            neuralNetwork.train(feature, label);
        }
    });

    console.log(i, neuralNetwork.calculateTotalError(trainings));

    trainings.sort(() => Math.random() - 0.5);
}

console.log(JSON.stringify(trainingData.map(([feature]) => argmax(neuralNetwork.feetForward(feature)))));

function argmax(vector: number[]): number {
    return vector.indexOf(Math.max(...vector));
}
