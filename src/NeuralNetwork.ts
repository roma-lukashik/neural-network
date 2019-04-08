import NeuronLayer from './NeuronLayer';
import { Optimizer } from './optimizers';
import { ILossFunction } from './loss-functions';

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

    public train(trainingInputs: number[], trainingOutputs: number[], optimizer: Optimizer, lossFunction: ILossFunction) {
        this.feetForward(trainingInputs);

        const deltas = optimizer(this.hiddenLayers, this.outputLayer, trainingOutputs, lossFunction.dx);

        this.layers.forEach((layer, i) => this.updateNeuronsWeights(layer, deltas[i]));
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

    public calculateTotalError(trainingSet: Array<[number[], number[]]>, lossFunction: ILossFunction) {
        const outputLayerNeurons = this.outputLayer.getNeurons();

        return trainingSet.reduce((error, [trainingInputs, trainingOutputs]) => {
            this.feetForward(trainingInputs);
            const err = trainingOutputs.reduce((sum, trainingOutput, i) => sum + lossFunction.fx(outputLayerNeurons[i].getOutput(), trainingOutput), 0);
            return error + err;
        }, 0) / trainingSet.length;
    }
}
