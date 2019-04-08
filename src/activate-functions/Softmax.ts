import * as vector from '../engine/VectorsOperators';

export function softmax(targetVector: number[]): number[] {
    const max = vector.maxElement(targetVector);
    const exps = targetVector.map((x) => Math.exp(x - max));
    const sum = vector.sumElements(exps);
    return exps.map((x) => x / sum);
}

export function dxSoftmax(targetVector: number[]): number[] {
    return softmax(targetVector).map((x) => x * (1 - x));
}