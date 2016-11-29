package jnet.algorithm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.data.Statistics;
import jnet.net.CostFunction;
import jnet.net.Layer;
import jnet.net.Matrix;
import jnet.net.Network;
import jnet.net.Vector;

public class StochasticGradientDescent implements LearningAlgorithm {

	private static Logger logger = Logger.getGlobal();
	
	private int numEpochs;
	private int batchSize;
	private double learningRate;
	//private double momentum;
	
	// use these to keep track of weight gradients for mini-batch
	private Map<Layer, List<Matrix>> weightGradients;
	private Map<Layer, List<Vector>> biasGradients;
	
	public StochasticGradientDescent(int numEpochs, int batchSize, double learningRate, double momentum) {
		this.numEpochs = numEpochs;
		this.batchSize = batchSize;
		this.learningRate = learningRate;
		//this.momentum = momentum;
		
		weightGradients = new HashMap<>();
		biasGradients = new HashMap<>();
	}
	
	@Override
	public void execute(Network network, DataSet trainingSet, DataSet validationSet, CostFunction costFunction)
	{
		assert (network != null);
		assert (trainingSet != null);
		assert (validationSet != null);
		assert (costFunction != null);
		
		logger.log(Level.INFO, "Training with stochastic gradient descent");
		for (int epoch = 0; epoch < numEpochs; ++epoch) {	
			logger.log(Level.INFO, String.format("%s %d", "Starting epoch ", epoch));
			trainingSet.shuffle();			
			List<DataSet> batches = trainingSet.getMiniBatches(batchSize);
			
			// replaced the earlier for loop and moved the gradient calculation to a lambda
			// referencing a method of the class - this will ease separating the back-prop
			// aspect of the algorithm from the stochastic gradient descent
			batches.forEach(
					batch -> { 
						logger.log(Level.FINE, String.format("Training minibatch..."));
							batch.getDataInstances().forEach(
								(instance) -> { calculateAndCaptureGradient(network, instance, costFunction); }
								);
						
							adjustWeightsAndBiases(network);
							clearGradients();
						}
					);
			
			// TODO - early stopping on successful validation
			logger.log(Level.INFO, "Evaluating current epoch");
			Statistics stats = network.validateOrTest(validationSet, costFunction);
			stats.print(false);
			
		}	
	}
	

	private void feedForward(Network network, DataInstance instance)
	{
		network.evaluate(instance);
	}
	
	private void backPropagate(Network network, DataInstance instance, CostFunction costFunction)
	{
		Layer outputLayer = network.getOutputLayer();
		//outputLayer.backPropagate(instance, costFunction);
		
		// output layer error
		outputLayer.setError(Vector.schurProduct(costFunction.costPrime(outputLayer.getActivation(), instance.getExpectedOutputs()),
				outputLayer.getActivationFunction().firstDerivative(outputLayer.getWeightedInput())));
		
		Layer next = outputLayer;
		Layer prev = outputLayer.getPrevious();
		while (prev != null && prev.getActivationFunction() != null) {
			prev.setError(Vector.schurProduct(Matrix.multiply(Matrix.transpose(next.getWeights()), next.getError()),
					prev.getActivationFunction().firstDerivative(prev.getWeightedInput())));
			//prev.backPropagate(next);
			next = prev;
			prev = next.getPrevious();
		}
	}
	
	private void calculateGradient(Network network)
	{
		Layer layer = network.getOutputLayer();
		
		while (layer != null && layer.getPrevious() != null) {
			
			if (weightGradients.containsKey(layer)) {
				weightGradients.get(layer).add(layer.getWeightGradient());
			} else {
				weightGradients.put(layer, new ArrayList<>());
				weightGradients.get(layer).add(layer.getWeightGradient());
			}
			
			if (biasGradients.containsKey(layer)) {
				biasGradients.get(layer).add(layer.getBiasGradient());
			} else {
				biasGradients.put(layer, new ArrayList<>());
				biasGradients.get(layer).add(layer.getBiasGradient());
			}
			
			//layer.addBiasGradient(layer.getBiasGradient());
			//layer.addWeightGradient(layer.getWeightGradient());
			layer = layer.getPrevious();
		}
	}
	
	private void adjustWeightsAndBiases(Network network) 
	{
		// adjust weights and biases
		Layer layer = network.getOutputLayer();
		while (layer != null && layer.getPrevious() != null) {
			adjustWeights(layer, learningRate, getMeanWeightGradient(layer));
			adjustBiases(layer, learningRate, getMeanBiasGradient(layer));
			layer = layer.getPrevious();
		}
	}
	
	private void clearGradients() 
	{
		weightGradients.clear();
		biasGradients.clear();
	}
	
	private Matrix getMeanWeightGradient(Layer layer)
	{
		Matrix sum = null;
		for (Matrix weightGradient : weightGradients.get(layer)) {
			if (sum == null) {
				sum = weightGradient;
			} else {
				sum = Matrix.add(sum, weightGradient);
			}
		}
		return Matrix.multiply(1.0/weightGradients.get(layer).size(), sum);
	}
	
	private Vector getMeanBiasGradient(Layer layer)
	{
		Vector sum = null;
		for (Vector biasGradient : biasGradients.get(layer)) {
			if (sum == null) {
				sum = biasGradient;
			} else {
				sum = Vector.add(sum, biasGradient);
			}
		}
		return Vector.multiply(1.0/biasGradients.get(layer).size(), sum);
	}
	
	private void adjustWeights(Layer layer, double learningRate, Matrix meanWeightGradient) {
		if (layer.getWeights() != null) {
		   layer.setWeights(Matrix.add(layer.getWeights(), Matrix.multiply(-learningRate, meanWeightGradient)));
		}
	}

	private void adjustBiases(Layer layer, double learningRate, Vector meanBiasGradient) {
		if (layer.getBiases() != null) {
			layer.setBiases(Vector.add(layer.getBiases(), Vector.multiply(-learningRate, meanBiasGradient)));
		}
	}
	
	public void calculateAndCaptureGradient(Network network, DataInstance instance, CostFunction costFunction)
	{
		feedForward(network, instance);
		backPropagate(network, instance, costFunction);
		calculateGradient(network);
	}
	
}
