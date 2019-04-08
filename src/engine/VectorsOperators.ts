import * as array from './ArrayOperators';

export type Vector = number[];

export function dot(vectorA: Vector, vectorB: Vector): number {
    return array.pair(vectorA, vectorB).reduce((sum, [x, y]) => sum + x * y, 0);
}

export function hadamard(vectorA: Vector, vectorB: Vector): Vector {
    return array.pair(vectorA, vectorB).map(([x, y]) => x * y);
}

export function scalar(vector: Vector, number: number): Vector {
    return vector.map((x) => x * number);
}

export function argSum(vector: Vector): number {
    return vector.reduce((a, b) => a + b);
}

export function argMean(vector: Vector): number {
    return argSum(vector) / vector.length;
}

export function maxElement(vector: Vector): number {
    return Math.max(...vector);
}