import { Vector } from '../engine/VectorsOperators';

export function relu(vector: Vector): Vector {
    return vector.map((x) => Math.max(0, x));
}

export function dxRelu(vector: Vector): Vector {
    return vector.map((x) => x > 0 ? 1 : 0);
}