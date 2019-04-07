export function sigmoid(vector: number[]): number[] {
    return vector.map((x) => 1 / (1 + Math.exp(-x)));
}

export function dxSigmoid(vector: number[]): number[] {
    return sigmoid(vector).map((x) => x * (1 - x));
}