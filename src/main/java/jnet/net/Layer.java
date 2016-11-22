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
	private ActivationFunction activationFunction;

	private Vector error;

	// keep track of gradients over a mini-batch of test data
	// use the mean of these x learning rate to adjust weights
	// and biases for the next batch
	//private ArrayList<Vector> biasGradients;
	//private ArrayList<Matrix> weightGradients;

	// Constructor for layer
	// Construct the layer and initialize the weights/biases to a random
	// Gaussian distribution
	// If the layer is an input later, we don't need weights or biases
	public Layer(int numNeurons, Layer previous, ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
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
		if (previous == null || activationFunction == null)
			return;

		setWeightedInput(previous.getActivation());
		setActivation(activationFunction.evaluate(getWeightedInput()));
	}

	public void backPropagate(DataInstance dataInstance, CostFunction costFunction) {
		if (previous == null || activationFunction == null)
			return;
		
		// output layer
		setError(Vector.schurProduct(costFunction.costPrime(getActivation(), dataInstance.getExpectedOutputs()),
				activationFunction.firstDerivative(getWeightedInput())));
	}

	private Vector getWeightedInput() {
		// TODO Auto-generated method stub
		return weightedInput;
	}

	public void backPropagate(Layer next) {
		if (previous == null || activationFunction == null)
			return;
		
		setError(Vector.schurProduct(Matrix.multiply(Matrix.transpose(next.getWeights()), next.getError()),
				activationFunction.firstDerivative(getWeightedInput())));
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
/*
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
*/
	public Vector getActivation() {
		return activation;
	}

	public void setActivation(Vector activation) {
		this.activation = activation;
	}

	private void setWeightedInput(Vector prevActivation) {
		weightedInput = Vector.add(Matrix.multiply(weights, prevActivation), biases);
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

	public void adjustWeights(double learningRate, Matrix meanWeightGradient) {
		if (weights != null) {
		   //assert (weightGradients != null && weightGradients.size() >  0);
		   //weights = Matrix.add(weights, Matrix.multiply(-learningRate/(weightGradients.size()), getSumOfWeightGradients()));
		   weights = Matrix.add(weights, Matrix.multiply(-learningRate, meanWeightGradient));
		}
	}
/*
	private Matrix getSumOfWeightGradients() {
		Matrix sum = new Matrix(weights.getNumRows(), weights.getNumCols());
		for (Matrix weightGradient : weightGradients) {
			sum = Matrix.add(sum, weightGradient);
		}
		return sum;
	}
*/
	public void adjustBiases(double learningRate, Vector meanBiasGradient) {
		if (biases != null) {
			//assert (biasGradients != null && biasGradients.size() > 0);
			//this.biases = Vector.add(this.biases, Vector.multiply(-learningRate/(biasGradients.size()), getSumOfBiasGradients()));
			this.biases = Vector.add(this.biases, Vector.multiply(-learningRate, meanBiasGradient));
		}
	}
/*
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
	*/
}
