export function relu(vector: number[]): number[] {
    return vector.map((x) => Math.max(0, x));
}

export function dxRelu(vector: number[]): number[] {
    return vector.map((x) => x > 0 ? 1 : 0);
}