import NeuronLayer from './NeuronLayer';
import { gradientDescent, Optimizer } from './optimizers';
import { ILossFunction, LossFunction, LossFunctions } from './loss-functions';
import * as array from './engine/ArrayOperators';
import * as vector from './engine/VectorsOperators';
import * as Distributions from './engine/Distributions';
import Vector = vector.Vector;
import Neuron from './Neuron';

interface INeuralNetworkOptions {
    learningRate: number;
    weightInitializationStrategy: () => number;
}

interface ITrainOptions {
    optimizer: Optimizer;
    lossFunction: ILossFunction;
    epochs: number;
}

const defaultNeuralNetworkOptions: INeuralNetworkOptions = {
    learningRate: 0.1,
    weightInitializationStrategy: Distributions.uniform,
};

const defaultTrainOptions: ITrainOptions = {
    optimizer: gradientDescent,
    lossFunction: LossFunctions.quadratic,
    epochs: 5,
};

export default class NeuralNetwork {
    private readonly hiddenLayers: NeuronLayer[];
    private readonly outputLayer: NeuronLayer;

    constructor(
        private readonly inputsNumber: number,
        private readonly layers: NeuronLayer[],
        private readonly neuralNetworkOptions: Partial<INeuralNetworkOptions> = {},
    ) {
        this.neuralNetworkOptions = { ...defaultNeuralNetworkOptions, ...this.neuralNetworkOptions };
        this.hiddenLayers = this.layers.slice(0, this.layers.length - 1);
        this.outputLayer = this.layers[this.layers.length - 1];

        this.initializeWeights();
    }

    private initializeWeights() {
        const [firstLayer, ...otherLayers] = this.layers;

        firstLayer.getNeurons().forEach((neuron) => {
            array.times(this.inputsNumber, () => this.initializeWeight(neuron));
        });

        otherLayers.forEach((layer, i) => {
            const previousLayer = this.layers[i];
            layer.getNeurons().forEach((neuron) => {
                previousLayer.getNeurons().forEach(() => this.initializeWeight(neuron));
            });
        });
    }

    private initializeWeight(neuron: Neuron) {
        neuron.addWeight(this.neuralNetworkOptions.weightInitializationStrategy());
    }

    public train(features: Vector[], labels: Vector[], trainOptions: Partial<ITrainOptions> = {}): Vector {
        const { optimizer, lossFunction, epochs } = { ...defaultTrainOptions, ...trainOptions };
        const trainingSet = array.pair(features, labels);
        const errors = [] as Vector;

        array.times(epochs, () => {
            let loss = 0;

            trainingSet.forEach(([feature, label]) => {
                // TODO fix that
                array.times(5, () => {
                    this.feetForward(feature);
                    const deltas = optimizer(this.hiddenLayers, this.outputLayer, label, lossFunction.dx);
                    this.layers.forEach((layer, i) => this.updateNeuronsWeights(layer, deltas[i]));
                });

                loss += this.calculateError(label, lossFunction.fx);
            });

            trainingSet.sort(() => Math.random() - 0.5);
            errors.push(loss / trainingSet.length);
        });

        return errors;
    }

    public feetForward(feature: Vector): Vector {
        let output = feature;

        this.layers.forEach((layer) => {
            output = layer.feedForward(output);
        });

        return output;
    }

    private updateNeuronsWeights(neuronLayer: NeuronLayer, deltas: Vector) {
        const deltasWithLearningRate = vector.scalar(deltas, this.neuralNetworkOptions.learningRate);

        neuronLayer.getNeurons().forEach((neuron, i) => {
            neuron.getWeights().forEach((neuronWeight, j) => {
                const weightDelta = deltasWithLearningRate[i] * neuron.calculatePdTotalNetInputWrtWeight(j);
                neuronWeight.setValue(neuronWeight.getValue() - weightDelta);
            });
            neuron.setBias(neuron.getBias() - deltasWithLearningRate[i]);
        });
    }

    private calculateError(label: Vector, lossFunction: LossFunction): number {
        const errors = array.pair(this.outputLayer.getOutput(), label).map(([prediction, target]) => {
            return lossFunction(prediction, target);
        });
        return vector.argSum(errors);
    }
}
