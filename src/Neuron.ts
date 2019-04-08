import Weight from './Weight';

export default class Neuron {
    private readonly weights: Weight[] = [];
    private inputs: number[] = [];
    private output: number = 0;

    constructor(private bias: number) { }

    public addWeight(weight: number) {
        this.weights.push(new Weight(weight));
    }

    public getWeights(): Weight[] {
        return this.weights;
    }

    public getWeight(index: number): Weight {
        return this.weights[index];
    }

    public calculatePdTotalNetInputWrtWeight(index: number): number {
        return this.inputs[index];
    }

    public setInputs(inputs: number[]) {
        this.inputs = inputs;
    }

    public getBias(): number {
        return this.bias;
    }

    public setBias(bias: number) {
        this.bias = bias;
    }

    public getOutput(): number {
        return this.output;
    }

    public calculateNeuronActivation(output: number): number {
        return this.output = output;
    }

    public calculateInputSum(): number {
        return this.inputs.reduce((sum, input, i) => sum + input * this.weights[i].getValue(), this.bias);
    }
}
