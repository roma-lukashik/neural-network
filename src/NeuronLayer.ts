import Neuron from './Neuron';
import { IActivateFunction } from './activate-functions';

export default class NeuronLayer {
    private readonly neurons: Neuron[];

    constructor(neuronsNumber: number, private activateFn: IActivateFunction, bias: number = Math.random() - 0.5) {
        this.neurons = Array(neuronsNumber).fill(0).map(() => new Neuron(bias));
    }

    public getNeurons(): Neuron[] {
        return this.neurons;
    }

    public feedForward(inputs: number[]): number[] {
        this.neurons.forEach((neuron) => neuron.setInputs(inputs));

        const outputs = this.calculateNeuronsActivation();

        return this.neurons.map((neuron, i) => neuron.calculateNeuronActivation(outputs[i]));
    }

    public calculateNeuronsActivation(): number[] {
        return this.activateFn.fx(this.getNeuronsInputSum());
    }

    public calculatePdTotalNetInputWrtInput(): number[] {
        return this.activateFn.dx(this.getNeuronsInputSum());
    }

    private getNeuronsInputSum(): number[] {
        return this.neurons.map((neuron) => neuron.calculateInputSum());
    }
}
