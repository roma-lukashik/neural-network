export function quadratic(prediction: number, target: number): number {
    return 0.5 * (target - prediction) ** 2;
}

export function dxQuadratic(prediction: number, target: number): number {
    return target - prediction;
}