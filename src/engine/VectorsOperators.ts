import * as array from './ArrayOperators';

export function dot(vectorA: number[], vectorB: number[]): number {
    return array.pair(vectorA, vectorB).reduce((sum, [x, y]) => sum + x * y, 0);
}

export function hadamard(vectorA: number[], vectorB: number[]): number[] {
    return array.pair(vectorA, vectorB).map(([x, y]) => x * y);
}

export function scalar(vector: number[], number: number): number[] {
    return vector.map((x) => x * number);
}

export function argSum(vector: number[]): number {
    return vector.reduce((a, b) => a + b);
}

export function argMean(vector: number[]): number {
    return argSum(vector) / vector.length;
}

export function maxElement(vector: number[]): number {
    return Math.max(...vector);
}