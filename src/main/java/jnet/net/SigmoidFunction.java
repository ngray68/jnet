package jnet.net;

import java.util.ArrayList;

public class SigmoidFunction implements ActivationFunction {

	@Override
	public double evaluate(double weightedInput) {
		return 1 / (1 + Math.exp(-weightedInput));
	}

	@Override
	public double firstDerivative(double weightedInput) {
		return evaluate(weightedInput) * (1 - evaluate(weightedInput));
	}

	@Override
	public Vector evaluate(Vector weightedInputs) {
		ArrayList<Double> result = new ArrayList<Double>();
		for (int i = 0; i < weightedInputs.getSize(); ++i) {
			result.add(i, evaluate(weightedInputs.getElement(i)));
		}
		return new Vector(result);
	}

	@Override
	public Vector firstDerivative(Vector weightedInputs) {
		ArrayList<Double> result = new ArrayList<Double>();
		for (int i = 0; i < weightedInputs.getSize(); ++i) {
			result.add(i, firstDerivative(weightedInputs.getElement(i)));
		}
		return new Vector(result);
	}

}
