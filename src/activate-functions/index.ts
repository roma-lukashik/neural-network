import { dxSigmoid, sigmoid } from './Sigmoid';
import { dxRelu, relu } from './Relu';
import { dxSoftmax, softmax } from './Softmax';
import { Vector } from '../engine/VectorsOperators';

export interface IActivateFunction {
    fx: (vector: Vector) => Vector,
    dx: (vector: Vector) => Vector,
}

export const ActivateFunctions = {
    sigmoid: { fx: sigmoid, dx: dxSigmoid },
    relu: { fx: relu, dx: dxRelu },
    softmax: { fx: softmax, dx: dxSoftmax },
};
