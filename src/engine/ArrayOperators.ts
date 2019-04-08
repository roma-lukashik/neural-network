export function pair<T, K>(arrayA: T[], arrayB: K[]): Array<[T, K]> {
    if (arrayA.length !== arrayB.length) {
        throw new Error('Arrays lengths are not equal.');
    }

    return arrayA.map<[T, K]>((item, i) => [item, arrayB[i]]);
}

export function times(count: number, callback: (i: number) => void): void {
    for (let i = 0; i < count; i++) {
        callback(i);
    }
}
