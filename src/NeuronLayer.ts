import Neuron from './Neuron';
import { IActivateFunction } from './activate-functions';
import * as array from './engine/ArrayOperators';
import { Vector } from './engine/VectorsOperators';

export default class NeuronLayer {
    private readonly neurons: Neuron[];

    constructor(neuronsNumber: number, private activateFn: IActivateFunction, bias: number = Math.random() - 0.5) {
        this.neurons = Array(neuronsNumber).fill(0).map(() => new Neuron(bias));
    }

    public getNeurons(): Neuron[] {
        return this.neurons;
    }

    public feedForward(inputs: Vector): Vector {
        this.neurons.forEach((neuron) => neuron.setInputs(inputs));

        const outputs = this.calculateNeuronsActivation();

        array.pair(this.neurons, outputs).forEach(([neuron, output]) => neuron.setOutput(output));

        return outputs;
    }

    public calculateNeuronsActivation(): Vector {
        return this.activateFn.fx(this.getNeuronsInputSum());
    }

    public calculatePdTotalNetInputWrtInput(): Vector {
        return this.activateFn.dx(this.getNeuronsInputSum());
    }

    private getNeuronsInputSum(): Vector {
        return this.neurons.map((neuron) => neuron.calculateInputSum());
    }
}
