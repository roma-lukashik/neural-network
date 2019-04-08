const NeuralNetwork = require('../../build/src/NeuralNetwork').default;
const NeuronLayer = require('../../build/src/NeuronLayer').default;
const { ActivateFunctions } = require('../../build/src/activate-functions/index');
const { features, labels } = require('./trainingSet');

const imageSize = 28 * 28;
const classesNumber = 10; // ten unique numbers

const layers = [
    new NeuronLayer(100, ActivateFunctions.sigmoid),
    new NeuronLayer(classesNumber, ActivateFunctions.softmax),
];

const neuralNetwork = new NeuralNetwork(imageSize, layers);
const errors = neuralNetwork.train(features, labels);

console.log(JSON.stringify(errors));
console.log(JSON.stringify(features.map((feature) => argmax(neuralNetwork.feetForward(feature)))));

function argmax(vector) {
    return vector.indexOf(Math.max(...vector));
}
