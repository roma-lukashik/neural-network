import NeuronLayer from '../NeuronLayer';

export type Optimizer = (
    hiddenLayers: NeuronLayer[],
    outputLayer: NeuronLayer,
    trainingOutputs: number[],
) => number[][];

export { gradientDescent } from './GradientDescent';
