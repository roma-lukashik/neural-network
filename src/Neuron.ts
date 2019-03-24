import Weight from './Weight';
import { IActivateFunction } from './ActivateFunctions';

export default class Neuron {
    private readonly weights: Weight[] = [];
    private inputs: number[] = [];
    private output: number = 0;

    constructor(
        private readonly activateFn: IActivateFunction,
        private readonly bias: number
    ) {
    }

    public addWeight(weight: number) {
        this.weights.push(new Weight(weight));
    }

    public getWeights(): Weight[] {
        return this.weights;
    }

    public getWeight(index: number): Weight {
        return this.weights[index];
    }

    public getInput(index: number): number {
        return this.inputs[index];
    }

    public calculateOutput(inputs: number[]): number {
        this.inputs = inputs;
        this.output = this.activateFn.fx(this.calculateTotalInput());
        return this.output;
    }

    private calculateTotalInput(): number {
        return this.inputs.reduce((sum, input, i) => sum + input * this.weights[i].getValue(), this.bias);
    }

    public calculateDelta(targetOutput: number): number {
        return this.calculateOutputError(targetOutput) * this.calculateDerivativeTotalInput();
    }

    private calculateOutputError(targetOutput: number): number {
        return -(targetOutput - this.output);
    }

    public calculateDerivativeTotalInput(): number {
        return this.activateFn.dx(this.calculateTotalInput());
    }

    public calculateError(targetOutput: number): number {
        return 0.5 * (targetOutput - this.output) ** 2;
    }
}
