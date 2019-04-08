import * as vector from '../engine/VectorsOperators';
import Vector = vector.Vector;

export function softmax(targetVector: Vector): Vector {
    const max = vector.maxElement(targetVector);
    const exps = targetVector.map((x) => Math.exp(x - max));
    const sum = vector.argSum(exps);
    return exps.map((x) => x / sum);
}

export function dxSoftmax(targetVector: Vector): Vector {
    return softmax(targetVector).map((x) => x * (1 - x));
}