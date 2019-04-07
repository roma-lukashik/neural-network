import { dxSigmoid, sigmoid } from './Sigmoid';
import { dxRelu, relu } from './Relu';
import { dxSoftmax, softmax } from './Softmax';

export interface IActivateFunction {
    fx: (vector: number[]) => number[],
    dx: (vector: number[]) => number[],
}

export class ActivateFunctions {
    public static sigmoid: IActivateFunction = {
        fx: sigmoid,
        dx: dxSigmoid,
    };

    public static relu: IActivateFunction = {
        fx: relu,
        dx: dxRelu,
    };

    public static softmax: IActivateFunction = {
        fx: softmax,
        dx: dxSoftmax,
    };
}
