import NeuronLayer from '../NeuronLayer';
import { LossFunction } from '../loss-functions';

export type Optimizer = (
    hiddenLayers: NeuronLayer[],
    outputLayer: NeuronLayer,
    trainingOutputs: number[],
    dxLossFunction: LossFunction,
) => number[][];

export { gradientDescent } from './GradientDescent';
