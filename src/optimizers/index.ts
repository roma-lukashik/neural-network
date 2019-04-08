import NeuronLayer from '../NeuronLayer';
import { LossFunction } from '../loss-functions';
import { Vector } from '../engine/VectorsOperators';

export type Optimizer = (
    hiddenLayers: NeuronLayer[],
    outputLayer: NeuronLayer,
    trainingOutputs: Vector,
    dxLossFunction: LossFunction,
) => Vector[];

export { gradientDescent } from './GradientDescent';
