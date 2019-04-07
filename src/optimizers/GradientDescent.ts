import NeuronLayer from '../NeuronLayer';

export function gradientDescent(hiddenLayers: NeuronLayer[], outputLayer: NeuronLayer, trainingOutputs: number[]): number[][] {
    const outputDeltas = calculateOutputDeltas(outputLayer, trainingOutputs);
    const hiddenDeltas = calculateHiddenDeltas(hiddenLayers, outputLayer, outputDeltas);

    return [...hiddenDeltas, outputDeltas];
}

function calculateOutputDeltas(outputLayer: NeuronLayer, trainingOutputs: number[]): number[] {
    const derivativeActivation = outputLayer.calculatePdTotalNetInputWrtInput();
    return calculatePdErrorWrtOutput(outputLayer, trainingOutputs).map((pdErrorWrtOutput, i) => {
        return pdErrorWrtOutput * derivativeActivation[i];
    });
}

function calculatePdErrorWrtOutput(outputLayer: NeuronLayer, trainingOutputs: number[]): number[] {
    return outputLayer.getNeurons().map((outputNeuron, i) => {
        return outputNeuron.getOutput() - trainingOutputs[i];
    });
}

function calculateHiddenDeltas(previousLayers: NeuronLayer[], nextLayer: NeuronLayer, nextLayerDeltas: number[]): number[][] {
    const previousLayersCopy = [...previousLayers];
    const previousLayer = previousLayersCopy.pop();
    const derivativeActivation = previousLayer.calculatePdTotalNetInputWrtInput();

    const previousLayerDeltas = previousLayer.getNeurons().map((previousNeuron, i) => {
        return derivativeActivation[i] * nextLayer.getNeurons().reduce((sum, nextNeuron, j) => {
            return sum + nextLayerDeltas[j] * nextNeuron.getWeight(i).getValue();
        }, 0);
    });

    if (previousLayersCopy.length > 0) {
        return [...calculateHiddenDeltas(previousLayersCopy, previousLayer, previousLayerDeltas), previousLayerDeltas];
    } else {
        return [previousLayerDeltas];
    }
}