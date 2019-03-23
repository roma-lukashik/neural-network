function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dxSigmoid(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

export interface IActivateFunction {
    fx: (x: number) => number,
    dx: (x: number) => number,
}

export enum ActivateFunction {
    Sigmoid,
}

export const ActiveFunctions = {
    [ActivateFunction.Sigmoid]: {
        fx: sigmoid,
        dx: dxSigmoid,
    } as IActivateFunction,
};
