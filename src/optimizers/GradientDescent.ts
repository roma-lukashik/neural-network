import NeuronLayer from '../NeuronLayer';
import { LossFunction } from '../loss-functions';
import * as vector from '../engine/VectorsOperators';
import * as array from '../engine/ArrayOperators';
import Vector = vector.Vector;

export function gradientDescent(
    hiddenLayers: NeuronLayer[],
    outputLayer: NeuronLayer,
    trainingOutputs: Vector,
    dxLossFunction: LossFunction,
): Vector[] {
    const outputDeltas = calculateOutputDeltas(outputLayer, trainingOutputs, dxLossFunction);
    const hiddenDeltas = calculateHiddenDeltas(hiddenLayers, outputLayer, outputDeltas);

    return [...hiddenDeltas, outputDeltas];
}

function calculateOutputDeltas(outputLayer: NeuronLayer, trainingOutputs: Vector, dxLossFunction: LossFunction): Vector {
    const pdErrorWrtOutput = calculatePdErrorWrtOutput(outputLayer, trainingOutputs, dxLossFunction);
    const pdTotalNetInputWrtInput = outputLayer.calculatePdTotalNetInputWrtInput();
    return vector.hadamard(pdErrorWrtOutput, pdTotalNetInputWrtInput);
}

function calculatePdErrorWrtOutput(outputLayer: NeuronLayer, trainingOutputs: Vector, dxLossFunction: LossFunction): Vector {
    return array.pair(outputLayer.getNeurons(), trainingOutputs).map(([outputNeuron, trainingOutput]) => {
        return -dxLossFunction(outputNeuron.getOutput(), trainingOutput);
    });
}

function calculateHiddenDeltas(previousLayers: NeuronLayer[], nextLayer: NeuronLayer, nextLayerDeltas: Vector): Vector[] {
    const previousLayersCopy = [...previousLayers];
    const previousLayer = previousLayersCopy.pop();
    const derivativeActivation = previousLayer.calculatePdTotalNetInputWrtInput();

    const weights = previousLayer.getNeurons().map((_, i) => {
        return vector.dot(
            nextLayerDeltas,
            nextLayer.getNeurons().map((nextNeuron) => nextNeuron.getWeight(i).getValue()),
        );
    });

    const previousLayerDeltas = vector.hadamard(derivativeActivation, weights);

    if (previousLayersCopy.length > 0) {
        return [...calculateHiddenDeltas(previousLayersCopy, previousLayer, previousLayerDeltas), previousLayerDeltas];
    } else {
        return [previousLayerDeltas];
    }
}