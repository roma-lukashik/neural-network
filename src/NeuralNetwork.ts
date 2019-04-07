import NeuronLayer from './NeuronLayer';

export default class NeuralNetwork {
    constructor(
        private readonly inputsNumber: number,
        private readonly hiddenLayers: NeuronLayer[],
        private readonly outputLayer: NeuronLayer,
        private readonly learningRate: number = 0.1,
    ) {
        this.initializeWeights();
    }

    private initializeWeights() {
        const [firstLayer, ...otherLayers] = this.hiddenLayers;

        // TODO refactor it
        firstLayer.getNeurons().forEach((neuron) => {
            Array(this.inputsNumber).fill(0).forEach(() => neuron.addWeight(Math.random() - 0.5));
        });

        [...otherLayers, this.outputLayer].forEach((layer, i) => {
            const previousLayer = this.hiddenLayers[i];
            layer.getNeurons().forEach((neuron) => {
                previousLayer.getNeurons().forEach(() => neuron.addWeight(Math.random() - 0.5));
            });
        });
    }

    public train(trainingInputs: number[], trainingOutputs: number[]) {
        this.feetForward(trainingInputs);

        const outputDeltas = this.calculateOutputDeltas(trainingOutputs);
        const hiddenDeltas = this.calculateHiddenDeltas(this.hiddenLayers, this.outputLayer, outputDeltas);

        this.updateNeuronsWeights(this.outputLayer, outputDeltas);
        this.hiddenLayers.forEach((hiddenLayer, i) => this.updateNeuronsWeights(hiddenLayer, hiddenDeltas[i]));
    }

    public feetForward(inputs: number[]): number[] {
        return [...this.hiddenLayers, this.outputLayer].reduce(
            (nextLayerInput, nextLayer) => nextLayer.feedForward(nextLayerInput),
            inputs,
        );
    }

    private calculateOutputDeltas(trainingOutputs: number[]): number[] {
        const derivativeActivation = this.outputLayer.calculatePdTotalNetInputWrtInput();
        return this.calculatePdErrorWrtOutput(trainingOutputs).map((pdErrorWrtOutput, i) => {
            return pdErrorWrtOutput * derivativeActivation[i];
        });
    }

    private calculatePdErrorWrtOutput(trainingOutputs: number[]) {
        return this.outputLayer.getNeurons().map((outputNeuron, i) => {
            return outputNeuron.getOutput() - trainingOutputs[i];
        });
    }

    private calculateHiddenDeltas(previousLayers: NeuronLayer[], nextLayer: NeuronLayer, nextLayerDeltas: number[]) {
        const previousLayersCopy = [...previousLayers];
        const previousLayer = previousLayersCopy.pop();
        const derivativeActivation = previousLayer.calculatePdTotalNetInputWrtInput();

        const previousLayerDeltas = previousLayer.getNeurons().map((previousNeuron, i) => {
            return derivativeActivation[i] * nextLayer.getNeurons().reduce((sum, nextNeuron, j) => {
                return sum + nextLayerDeltas[j] * nextNeuron.getWeight(i).getValue();
            }, 0);
        });

        if (previousLayersCopy.length > 0) {
            return [...this.calculateHiddenDeltas(previousLayersCopy, previousLayer, previousLayerDeltas), previousLayerDeltas];
        } else {
            return [previousLayerDeltas];
        }
    }

    private updateNeuronsWeights(neuronLayer: NeuronLayer, deltas: number[]) {
        neuronLayer.getNeurons().forEach((neuron, i) => {
            neuron.getWeights().forEach((neuronWeight, j) => {
                neuronWeight.setValue(neuronWeight.getValue() - this.learningRate * deltas[i] * neuron.calculatePdTotalNetInputWrtWeight(j));
            });
            neuron.setBias(neuron.getBias() - this.learningRate * deltas[i]);
        });
    }

    public calculateTotalError(trainingSet: Array<[number[], number[]]>) {
        const outputLayerNeurons = this.outputLayer.getNeurons();

        return trainingSet.reduce((error, [trainingInputs, trainingOutputs]) => {
            this.feetForward(trainingInputs);
            const err = trainingOutputs.reduce((sum, trainingOutput, i) => sum + outputLayerNeurons[i].calculateError(trainingOutput), 0);
            return error + err;
        }, 0) / trainingSet.length;
    }
}
