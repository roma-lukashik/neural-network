export default class Weight {
    constructor(private value: number) {
    }

    public getValue(): number {
        return this.value;
    }

    public setValue(value: number) {
        this.value = value;
    }
}
