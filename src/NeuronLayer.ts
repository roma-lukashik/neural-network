import Neuron from './Neuron';
import { IActivateFunction } from './ActivateFunctions';

export default class NeuronLayer {
    private readonly neurons: Neuron[];

    constructor(neuronsNumber: number, activateFn: IActivateFunction, bias: number = Math.random()) {
        this.neurons = Array(neuronsNumber).fill(0).map(() => new Neuron(activateFn, bias));
    }

    public getNeurons(): Neuron[] {
        return this.neurons;
    }

    public feedForward(inputs: number[]): number[] {
        return this.neurons.map((neuron) => neuron.calculateOutput(inputs));
    }
}
