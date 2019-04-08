import * as array from './ArrayOperators';

export function dot(vectorA: number[], vectorB: number[]): number {
    return array.pair(vectorA, vectorB).reduce((sum, [x, y]) => sum + x * y, 0);
}

export function hadamard(vectorA: number[], vectorB: number[]): number[] {
    return array.pair(vectorA, vectorB).map(([x, y]) => x * y);
}

export function sumElements(vector: number[]): number {
    return vector.reduce((a, b) => a + b);
}

export function meanElements(vector: number[]): number {
    return sumElements(vector) / vector.length;
}

export function maxElement(vector: number[]): number {
    return Math.max(...vector);
}