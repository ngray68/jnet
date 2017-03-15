package jnet.data;

import java.util.List;

import com.ngray.jnet.algebra.Vector;

public class DataInstance {
	
	private Vector inputs;
	private Vector expectedOutputs;
	
	public DataInstance(List<Double> inputs, List<Double> expectedOutputs) {
		this.inputs = new Vector(inputs);
		this.expectedOutputs = new Vector(expectedOutputs);
	}
	
	public DataInstance(Vector inputs, Vector expectedOutputs) {
		this.inputs = inputs;
		this.expectedOutputs = expectedOutputs;
	}

	public Vector getInputs() {
		return inputs;
	}
	
	public Vector getExpectedOutputs() {
		return expectedOutputs;
	}

	public void normalize(double[] minValues, double[] maxValues) {
		double[] normalizedInputs = new double[inputs.getSize()];
		for (int i = 0; i < inputs.getSize(); ++i) {
			if (maxValues[i] - minValues[i] != 0) {
				normalizedInputs[i] = (inputs.getElement(i) - minValues[i])/(maxValues[i] - minValues[i]);
			} else {
				normalizedInputs[i] = 0.0;
			}
		}
		inputs = new Vector(normalizedInputs);
	}
}
