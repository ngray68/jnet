package jnet.net;

import java.util.Random;

import com.ngray.jnet.algebra.Matrix;
import com.ngray.jnet.algebra.Vector;

public class Layer 
{

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
	public Layer(int numNeurons, Layer previous, ActivationFunction activationFunction) 
	{
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
	
	public int getNumNeurons() 
	{
		return numNeurons;
	}

	public Vector getWeightedInput() 
	{
		return weightedInput;
	}
	
	public void setWeightedInput(Vector input) throws NetworkException 
	{
		if (weights.getNumCols() != input.getSize()) {
			throw new NetworkException(String.format("Layer expects %d inputs, got %d", weights.getNumCols(), input.getSize()));
		}
		//weightedInput = Vector.add(Matrix.multiply(weights, input), biases);
		weightedInput = weights.multiply(input).add(biases);
	}

	public Matrix getWeights() 
	{	
		return weights;
	}
	
	public void setWeights(Matrix weights) throws NetworkException
	{
		if (weights.getNumRows() != getNumNeurons()) {
			throw new NetworkException(String.format("Attempt to set Layer weight matrix with %d rows inconsistent with number of neurons %d in the layer",
					weights.getNumRows(), getNumNeurons()));
		}
		this.weights = weights;
	}
	
	public Vector getBiases()
	{
		return biases;
	}
	
	public void setBiases(Vector biases) throws NetworkException
	{
		if (biases.getSize() != getNumNeurons()) {
			throw new NetworkException(String.format("Attempt to set Layer bias vector of size %d inconsistent with number of neurons %d in the layer",
					weights.getNumRows(), getNumNeurons()));
		}
		this.biases = biases;
	}

	public Vector getBiasGradient() 
	{
		return error;
	}

	public Matrix getWeightGradient() 
	{
		return error.dyadicProduct(previous.getActivation());
	}

	public Vector getActivation() 
	{
		return activation;
	}

	public void setActivation(Vector activation) 
	{
		this.activation = activation;
	}

	public Layer getPrevious() 
	{
		return previous;
	}

	public void setPrevious(Layer previous) 
	{
		this.previous = previous;
	}

	public Vector getError()
	{
		return error;
	}

	public void setError(Vector error) 
	{
		this.error = error;
	}

	public ActivationFunction getActivationFunction() 
	{
		return activationFunction;
	}
	
	public void setActivationFunction(ActivationFunction activationFunction) 
	{
		this.activationFunction = activationFunction;
	}
}