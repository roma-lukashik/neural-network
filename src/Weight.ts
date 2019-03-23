export default class Weight {
    constructor(private value: number) {
    }

    public getValue(): number {
        return this.value;
    }

    public setValue(value) {
        this.value = value;
    }
}
