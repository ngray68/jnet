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

	public void backPropagate(DataInstance dataInstance, CostFunction costFunction) 
	{
		if (previous == null || activationFunction == null)
			return;
		
		// output layer
		setError(Vector.schurProduct(costFunction.costPrime(getActivation(), dataInstance.getExpectedOutputs()),
				activationFunction.firstDerivative(getWeightedInput())));
	}

	private Vector getWeightedInput() 
	{
		// TODO Auto-generated method stub
		return weightedInput;
	}

	public void backPropagate(Layer next) 
	{
		if (previous == null || activationFunction == null)
			return;
		
		setError(Vector.schurProduct(Matrix.multiply(Matrix.transpose(next.getWeights()), next.getError()),
				activationFunction.firstDerivative(getWeightedInput())));
	}

	public Matrix getWeights() 
	{	
		return weights;
	}
	
	public void setWeights(Matrix weights)
	{
		this.weights = weights;
	}
	
	public Vector getBiases()
	{
		return biases;
	}
	
	public void setBiases(Vector biases)
	{
		this.biases = biases;
	}

	public Vector getBiasGradient() 
	{
		return error;
	}

	public Matrix getWeightGradient() 
	{
		return Vector.dyadicProduct(error, previous.getActivation());
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
	
/* Moved to stochastic gradient descent class
	public void adjustWeights(double learningRate, Matrix meanWeightGradient) {
		if (weights != null) {
		   weights = Matrix.add(weights, Matrix.multiply(-learningRate, meanWeightGradient));
		}
	}

	public void adjustBiases(double learningRate, Vector meanBiasGradient) {
		if (biases != null) {
			this.biases = Vector.add(this.biases, Vector.multiply(-learningRate, meanBiasGradient));
		}
	}
	*/
}
