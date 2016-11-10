package jnet.data;

import java.util.List;

import jnet.net.Vector;

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

	public void normalize(Double[] minValues, Double[] maxValues) {
		Double[] normalizedInputs = new Double[inputs.getSize()];
		for (int i = 0; i < inputs.getSize(); ++i) {
			normalizedInputs[i] = (inputs.getElement(i) - minValues[i])/(maxValues[i] - minValues[i]);
		}
		inputs = new Vector(normalizedInputs);
	}
}