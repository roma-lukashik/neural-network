import NeuralNetwork from '../NeuralNetwork';
import NeuronLayer from '../NeuronLayer';
import { trainingData } from './trainExamples';
import { ActivateFunctions } from '../activate-functions';
import { gradientDescent } from '../optimizers';
import { LossFunctions } from '../loss-functions';

const imageSize = 28 * 28;
const classesNumber = 10; // ten unique numbers

const layers = [
    new NeuronLayer(100, ActivateFunctions.sigmoid),
    new NeuronLayer(classesNumber, ActivateFunctions.softmax),
];

const neuralNetwork = new NeuralNetwork(imageSize, layers);

const trainings = [...trainingData];

const epochs = 5;

for (let i = 1; i <= epochs; i++) {
    trainings.forEach(([feature, label]) => {
        for (let j = 0; j < 5; j++) {
            neuralNetwork.train(feature, label, gradientDescent, LossFunctions.quadratic);
        }
    });

    console.log(i, neuralNetwork.calculateTotalError(trainings, LossFunctions.quadratic));

    trainings.sort(() => Math.random() - 0.5);
}

console.log(JSON.stringify(trainingData.map(([feature]) => argmax(neuralNetwork.feetForward(feature)))));

function argmax(vector: number[]): number {
    return vector.indexOf(Math.max(...vector));
}
