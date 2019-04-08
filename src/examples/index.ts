import NeuralNetwork from '../NeuralNetwork';
import NeuronLayer from '../NeuronLayer';
import { features, labels } from './trainExamples';
import { ActivateFunctions } from '../activate-functions';

const imageSize = 28 * 28;
const classesNumber = 10; // ten unique numbers

const layers = [
    new NeuronLayer(100, ActivateFunctions.sigmoid),
    new NeuronLayer(classesNumber, ActivateFunctions.softmax),
];

const neuralNetwork = new NeuralNetwork(imageSize, layers);
neuralNetwork.train(features, labels);

console.log(JSON.stringify(features.map((feature) => argmax(neuralNetwork.feetForward(feature)))));

function argmax(vector: number[]): number {
    return vector.indexOf(Math.max(...vector));
}
