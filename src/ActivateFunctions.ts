function sigmoid(vector: number[]): number[] {
    return vector.map((x) => 1 / (1 + Math.exp(-x)));
}

function dxSigmoid(vector: number[]): number[] {
    return sigmoid(vector).map((x) => x * (1 - x));
}

function relu(vector: number[]): number[] {
    return vector.map((x) => Math.max(0, x));
}

function dxRelu(vector: number[]): number[] {
    return vector.map((x) => x > 0 ? 1 : 0);
}

function softmax(vector: number[]): number[] {
    const max = Math.max(...vector);
    const exps = vector.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sum);
}

function dxSoftmax(vector: number[]): number[] {
    return softmax(vector).map((x) => x * (1 - x));
}

export interface IActivateFunction {
    fx: (vector: number[]) => number[],
    dx: (vector: number[]) => number[],
}

export enum ActivateFunction {
    Sigmoid,
    Relu,
    Softmax,
}

export const ActiveFunctions = {
    [ActivateFunction.Sigmoid]: {
        fx: sigmoid,
        dx: dxSigmoid,
    } as IActivateFunction,
    [ActivateFunction.Relu]: {
        fx: relu,
        dx: dxRelu,
    } as IActivateFunction,
    [ActivateFunction.Softmax] : {
        fx: softmax,
        dx: dxSoftmax,
    } as IActivateFunction,
};
