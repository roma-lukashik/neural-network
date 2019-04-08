import NeuronLayer from './NeuronLayer';
import { gradientDescent, Optimizer } from './optimizers';
import { ILossFunction, LossFunction, LossFunctions } from './loss-functions';
import * as array from './engine/ArrayOperators';
import * as vector from './engine/VectorsOperators';

interface ITrainOptions {
    optimizer: Optimizer;
    lossFunction: ILossFunction;
    epochs: number;
}

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
        private readonly learningRate: number = 0.1,
    ) {
        this.hiddenLayers = this.layers.slice(0, this.layers.length - 1);
        this.outputLayer = this.layers[this.layers.length - 1];

        this.initializeWeights();
    }

    private initializeWeights() {
        const [firstLayer, ...otherLayers] = this.layers;

        firstLayer.getNeurons().forEach((neuron) => {
            Array(this.inputsNumber).fill(0).forEach(() => neuron.addWeight(Math.random() - 0.5));
        });

        otherLayers.forEach((layer, i) => {
            const previousLayer = this.layers[i];
            layer.getNeurons().forEach((neuron) => {
                previousLayer.getNeurons().forEach(() => neuron.addWeight(Math.random() - 0.5));
            });
        });
    }

    public train(features: number[][], labels: number[][], trainOptions: Partial<ITrainOptions> = defaultTrainOptions) {
        const { optimizer, lossFunction, epochs } = { ...defaultTrainOptions, ...trainOptions };
        const trainingSet = array.pair(features, labels);
        const errors = [] as number[][];

        for (let i = 0; i < epochs; i++) {
            errors[i] = [];

            trainingSet.forEach(([feature, label]) => {
                // TODO fix that
                for (let j = 0; j < 5; j++) {
                    this.feetForward(feature);

                    const deltas = optimizer(this.hiddenLayers, this.outputLayer, label, lossFunction.dx);

                    this.layers.forEach((layer, i) => this.updateNeuronsWeights(layer, deltas[i]));
                }

                errors[i].push(this.calculateError(label, lossFunction.fx));
            });

            console.log(i + 1, vector.meanElements(errors[i]));

            trainingSet.sort(() => Math.random() - 0.5);
        }
    }

    public feetForward(feature: number[]): number[] {
        return this.layers.reduce((layerInput, layer) => layer.feedForward(layerInput), feature);
    }

    private updateNeuronsWeights(neuronLayer: NeuronLayer, deltas: number[]) {
        neuronLayer.getNeurons().forEach((neuron, i) => {
            neuron.getWeights().forEach((neuronWeight, j) => {
                neuronWeight.setValue(neuronWeight.getValue() - this.learningRate * deltas[i] * neuron.calculatePdTotalNetInputWrtWeight(j));
            });
            neuron.setBias(neuron.getBias() - this.learningRate * deltas[i]);
        });
    }

    private calculateError(label: number[], lossFunction: LossFunction): number {
        const errors = array.pair(this.outputLayer.getNeurons(), label).map(([neuron, target]) => {
            return lossFunction(neuron.getOutput(), target);
        });
        return vector.sumElements(errors);
    }
}
