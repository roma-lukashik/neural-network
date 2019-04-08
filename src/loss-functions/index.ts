import { dxQuadratic, quadratic } from './Quadratic';

export type LossFunction = (prediction: number, target: number) => number;

export interface ILossFunction {
    fx: LossFunction;
    dx: LossFunction;
}

export const LossFunctions = {
    quadratic: { fx: quadratic, dx: dxQuadratic },
};
