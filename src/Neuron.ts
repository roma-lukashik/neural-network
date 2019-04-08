import Weight from './Weight';
import * as vector from './engine/VectorsOperators';
import Vector = vector.Vector;

export default class Neuron {
    private readonly weights: Weight[] = [];
    private inputs: Vector = [];
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

    public setInputs(inputs: Vector) {
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

    public setOutput(output: number) {
        this.output = output;
    }

    public calculateInputSum(): number {
        const weights = this.weights.map((weight) => weight.getValue());
        return this.bias + vector.dot(this.inputs, weights);
    }
}
