import NeuronLayer from './NeuronLayer';

export default class NeuralNetwork {
    constructor(
        private readonly numInputs: number,
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
            Array(this.numInputs).fill(0).forEach(() => neuron.addWeight(Math.random()));
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

        const outputDeltas = this.outputLayer.getNeurons().map((outputNeuron, i) => {
            return outputNeuron.calculateDelta(trainingOutputs[i]);
        });

        this.updateNeuronsWeights(this.outputLayer, outputDeltas);

        let nextLayer = this.outputLayer;
        let nextLayerDeltas = outputDeltas;

        [...this.hiddenLayers].reverse().forEach((previousLayer) => {
            const previousLayerDeltas = previousLayer.getNeurons().map((previousNeuron, i) => {
                const currentLayerOutputDeltas = nextLayer.getNeurons().reduce((sum, nextNeuron, j) => {
                    return sum + nextLayerDeltas[j] * nextNeuron.getWeight(i).getValue();
                }, 0);

                return currentLayerOutputDeltas * previousNeuron.calculateDerivativeTotalInput();
            });

            nextLayer = previousLayer;
            nextLayerDeltas = previousLayerDeltas;

            this.updateNeuronsWeights(previousLayer, previousLayerDeltas);
        });
    }

    private feetForward(inputs: number[]): number[] {
        const hiddenLayerOutputs = this.hiddenLayers.reduce((previousLayerOutputs, nextLayer) => nextLayer.feedForward(previousLayerOutputs), inputs);
        return this.outputLayer.feedForward(hiddenLayerOutputs);
    }

    private updateNeuronsWeights(neuronLayer: NeuronLayer, deltas: number[]) {
        neuronLayer.getNeurons().forEach((neuron, i) => {
            neuron.getWeights().forEach((neuronWeight, j) => {
                neuronWeight.setValue(neuronWeight.getValue() - this.learningRate * deltas[i] * neuron.getInput(j));
            });
        });
    }

    public calculateTotalError(trainingSet: Array<[number[], number[]]>) {
        const outputLayerNeurons = this.outputLayer.getNeurons();

        trainingSet.forEach(([trainingInputs]) => this.feetForward(trainingInputs));

        return trainingSet.reduce((error, [, trainingOutputs]) => {
            return trainingOutputs.reduce((sum, trainingOutput, i) => sum + outputLayerNeurons[i].calculateError(trainingOutput), error);
        }, 0);
    }
}
