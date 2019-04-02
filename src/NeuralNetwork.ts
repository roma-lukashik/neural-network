import NeuronLayer from './NeuronLayer';

export default class NeuralNetwork {
    constructor(
        private readonly inputsNumber: number,
        private readonly hiddenLayers: NeuronLayer[],
        private readonly outputLayer: NeuronLayer,
        private readonly learningRate: number = 0.5,
    ) {
        this.initializeWeights();
    }

    private initializeWeights() {
        const [firstLayer, ...otherLayers] = this.hiddenLayers;

        // TODO refactor it
        firstLayer.getNeurons().forEach((neuron) => {
            Array(this.inputsNumber).fill(0).forEach(() => neuron.addWeight(Math.random()));
        });

        [...otherLayers, this.outputLayer].forEach((layer, i) => {
            const previousLayer = this.hiddenLayers[i];
            layer.getNeurons().forEach((neuron) => {
                previousLayer.getNeurons().forEach(() => neuron.addWeight(Math.random()));
            });
        });
    }

    public train(trainingInputs: number[], trainingOutputs: number[]) {
        this.feetForward(trainingInputs);

        const outputDeltas = this.calculateOutputDeltas(trainingOutputs);

        // console.log(JSON.stringify(this.outputLayer.getNeurons().map((n) => n.getOutput())));

        this.updateNeuronsWeights(this.outputLayer, outputDeltas);
        this.updateHiddenLayersWeights(this.hiddenLayers, this.outputLayer, outputDeltas);
    }

    private feetForward(inputs: number[]): number[] {
        const hiddenLayerOutputs = this.hiddenLayers.reduce(
            (previousLayerOutputs, nextLayer) => nextLayer.feedForward(previousLayerOutputs),
            inputs,
        );

        return this.outputLayer.feedForward(hiddenLayerOutputs);
    }

    private calculateOutputDeltas(trainingOutputs: number[]): number[] {
        const derivativeActivation = this.outputLayer.calculateDerivativeNeuronsActivation();
        return this.calculateDerivativeCrossEntropy(trainingOutputs).map((derivativeCrossEntropy, i) => {
            return derivativeCrossEntropy * derivativeActivation[i];
        });
    }

    private calculateDerivativeCrossEntropy(trainingOutputs: number[]) {
        return this.outputLayer.getNeurons().map((outputNeuron, i) => {
            return -(
                trainingOutputs[i] * (1 / outputNeuron.getOutput()) +
                (1 - trainingOutputs[i]) * (1 / (1 - outputNeuron.getOutput()))
            );
        });
    }

    private updateNeuronsWeights(neuronLayer: NeuronLayer, deltas: number[]) {
        neuronLayer.getNeurons().forEach((neuron, i) => {
            neuron.getWeights().forEach((neuronWeight, j) => {
                neuronWeight.setValue(neuronWeight.getValue() + this.learningRate * deltas[i] * neuron.getInput(j));
            });
        });
    }

    private updateHiddenLayersWeights(previousLayers: NeuronLayer[], nextLayer: NeuronLayer, nextLayerDeltas: number[]) {
        const previousLayersCopy = [...previousLayers];
        const previousLayer = previousLayersCopy.pop();
        const derivativeActivation = previousLayer.calculateDerivativeNeuronsActivation();

        const previousLayerDeltas = previousLayer.getNeurons().map((previousNeuron, i) => {
            return derivativeActivation[i] * nextLayer.getNeurons().reduce((sum, nextNeuron, j) => {
                return sum + nextLayerDeltas[j] * nextNeuron.getWeight(i).getValue();
            }, 0);
        });

        this.updateNeuronsWeights(previousLayer, previousLayerDeltas);

        if (previousLayersCopy.length > 0) {
            this.updateHiddenLayersWeights(previousLayersCopy, previousLayer, previousLayerDeltas);
        }
    }

    public calculateTotalError(trainingSet: Array<[number[], number[]]>) {
        const outputLayerNeurons = this.outputLayer.getNeurons();

        trainingSet.forEach(([trainingInputs]) => this.feetForward(trainingInputs));

        return trainingSet.reduce((error, [, trainingOutputs]) => {
            return trainingOutputs.reduce(
                (sum, trainingOutput, i) => sum + outputLayerNeurons[i].calculateError(trainingOutput),
                error,
            );
        }, 0);
    }
}
