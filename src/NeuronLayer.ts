import Neuron from './Neuron';
import { IActivateFunction } from './ActivateFunctions';

export default class NeuronLayer {
    private readonly neurons: Neuron[];

    constructor(numNeurons: number, activateFn: IActivateFunction, bias: number = Math.random()) {
        this.neurons = Array(numNeurons).fill(0).map(() => new Neuron(activateFn, bias));
    }

    public getNeurons(): Neuron[] {
        return this.neurons;
    }

    public getOutputs(): number[] {
        return this.neurons.map((neuron) => neuron.getOutput());
    }

    public feedForward(inputs: number[]): number[] {
        return this.neurons.map((neuron) => neuron.calculateOutput(inputs));
    }
}
