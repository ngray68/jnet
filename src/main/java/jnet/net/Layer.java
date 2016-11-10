package jnet.net;

import java.util.ArrayList;
import java.util.Random;

import jnet.data.DataInstance;

public class Layer {

	private int numNeurons;
	private Layer previous;
	private Matrix weights;
	private Vector biases;
	private Vector weightedInput;
	private Vector activation;

	private Vector error;

	// keep track of gradients over a mini-batch of test data
	// use the mean of these x learning rate to adjust weights
	// and biases for the next batch
	private ArrayList<Vector> biasGradients;
	private ArrayList<Matrix> weightGradients;

	// Constructor for layer
	// Construct the layer and initialize the weights/biases to a random
	// Gaussian distribution
	// If the layer is an input later, we don't need weights or biases
	public Layer(int numNeurons, Layer previous) {
		this.numNeurons = numNeurons;
		this.previous = previous;
		if (previous != null) {
			weights = new Matrix(numNeurons, previous.getNumNeurons(), new Random());
			biases = new Vector(numNeurons, new Random());
		} else {
			weights = null;
			biases = null;
		}
	}

	public int getNumNeurons() {
		return numNeurons;
	}

	public void feedForward() {
		if (previous == null)
			return;

		setWeightedInput(previous.getActivation());
		setActivation(sigmoid(getWeightedInput()));
	}

	public void backPropagate(DataInstance dataInstance, CostFunction costFunction) {
		if (previous == null)
			return;
		
		// output layer
		setError(Vector.schurProduct(costFunction.costPrime(getActivation(), dataInstance.getExpectedOutputs()),
				sigmoidPrime(getWeightedInput())));
	}

	private Vector getWeightedInput() {
		// TODO Auto-generated method stub
		return weightedInput;
	}

	public void backPropagate(Layer next) {
		if (previous == null)
			return;
		
		setError(Vector.schurProduct(Matrix.multiply(Matrix.transpose(next.getWeights()), next.getError()),
				sigmoidPrime(getWeightedInput())));
	}

	public Matrix getWeights() {
		// TODO Auto-generated method stub
		return weights;
	}

	public Vector getBiasGradient() {
		return error;
	}

	public Matrix getWeightGradient() {
		return Vector.dyadicProduct(error, previous.getActivation());
	}

	public void addBiasGradient(Vector biasGradient) {
		if (biasGradients == null)
			biasGradients = new ArrayList<Vector>();

		biasGradients.add(biasGradient);
	}

	public void addWeightGradient(Matrix weightGradient) {
		if (weightGradients == null)
			weightGradients = new ArrayList<Matrix>();
		
		weightGradients.add(weightGradient);
	}

	public Vector getActivation() {
		return activation;
	}

	public void setActivation(Vector activation) {
		this.activation = activation;
	}

	private void setWeightedInput(Vector prevActivation) {
		weightedInput = Vector.add(Matrix.multiply(weights, prevActivation), biases);
	}

	private Vector sigmoid(Vector v) {
		ArrayList<Double> result = new ArrayList<Double>();
		for (int i = 0; i < v.getSize(); ++i) {
			result.add(i, sigmoid(v.getElement(i)));
		}
		return new Vector(result);
	}

	private Vector sigmoidPrime(Vector v) {
		ArrayList<Double> result = new ArrayList<Double>();
		for (int i = 0; i < v.getSize(); ++i) {
			result.add(i, sigmoidPrime(v.getElement(i)));
		}
		return new Vector(result);
	}

	private double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	private double sigmoidPrime(double z) {
		return sigmoid(z) * (1 - sigmoid(z));
	}

	public Layer getPrevious() {
		return previous;
	}

	public void setPrevious(Layer previous) {
		this.previous = previous;
	}

	public Vector getError() {
		return error;
	}

	private void setError(Vector error) {
		this.error = error;
	}

	public void adjustWeights(double learningRate) {
		if (weights != null) {
		   assert (weightGradients != null && weightGradients.size() >  0);
		   weights = Matrix.add(weights, Matrix.multiply(-learningRate/(weightGradients.size()), getSumOfWeightGradients()));
		}
	}

	private Matrix getSumOfWeightGradients() {
		Matrix sum = new Matrix(weights.getNumRows(), weights.getNumCols());
		for (Matrix weightGradient : weightGradients) {
			sum = Matrix.add(sum, weightGradient);
		}
		return sum;
	}

	public void adjustBiases(double learningRate) {
		if (biases != null) {
			assert (biasGradients != null && biasGradients.size() > 0);
			this.biases = Vector.add(this.biases, Vector.multiply(-learningRate/(biasGradients.size()), getSumOfBiasGradients()));
		}
	}

	private Vector getSumOfBiasGradients() {
		Vector sum = new Vector(biases.getSize());
		for (Vector biasGradient : biasGradients) {
			sum = Vector.add(sum, biasGradient);
		}
		return sum;
	}

	public void clearWeightGradients() {
		this.weightGradients = null;
	}

	public void clearBiasGradients() {
		this.biasGradients = null;
	}
}
