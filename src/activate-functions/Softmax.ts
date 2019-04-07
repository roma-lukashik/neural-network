export function softmax(vector: number[]): number[] {
    const max = Math.max(...vector);
    const exps = vector.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sum);
}

export function dxSoftmax(vector: number[]): number[] {
    return softmax(vector).map((x) => x * (1 - x));
}