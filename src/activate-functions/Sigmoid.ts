import { Vector } from '../engine/VectorsOperators';

export function sigmoid(vector: Vector): Vector {
    return vector.map((x) => 1 / (1 + Math.exp(-x)));
}

export function dxSigmoid(vector: Vector): Vector {
    return sigmoid(vector).map((x) => x * (1 - x));
}