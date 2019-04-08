import { dxSigmoid, sigmoid } from './Sigmoid';
import { dxRelu, relu } from './Relu';
import { dxSoftmax, softmax } from './Softmax';

export interface IActivateFunction {
    fx: (vector: number[]) => number[],
    dx: (vector: number[]) => number[],
}

export const ActivateFunctions = {
    sigmoid: { fx: sigmoid, dx: dxSigmoid },

    relu: { fx: relu, dx: dxRelu },

    softmax: { fx: softmax, dx: dxSoftmax },
};
