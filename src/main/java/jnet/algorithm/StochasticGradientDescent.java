package jnet.algorithm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.ngray.jnet.algebra.Matrix;
import com.ngray.jnet.algebra.Vector;

import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.data.Statistics;
import jnet.net.CostFunction;
import jnet.net.Layer;
import jnet.net.Network;
import jnet.net.NetworkException;

public class StochasticGradientDescent implements LearningAlgorithm {

	private static Logger logger = Logger.getGlobal();
	
	private int numEpochs;
	private int batchSize;
	private double learningRate;
	private double momentum;
	
	private int epoch;
	
	// use these to keep track of deltas for momentum calculation
	private Map<Layer, Matrix> prevWeightDelta;
	
	// optimize keeping track of weight gradients by summing as we go
	private Map<Layer, Matrix> sumOfWeightGradients;
	private Map<Layer, Vector> sumOfBiasGradients;

	private int batchNo;
	
	public StochasticGradientDescent(int numEpochs, int batchSize, double learningRate, double momentum)
	{
		this.numEpochs = numEpochs;
		this.batchSize = batchSize;
		this.learningRate = learningRate;
		this.momentum = momentum;
		
		prevWeightDelta = new HashMap<>();
		
		sumOfWeightGradients = new HashMap<>();
		sumOfBiasGradients = new HashMap<>();
	}
	
	@Override
	public void execute(Network network, DataSet trainingSet, DataSet validationSet, CostFunction costFunction) throws NetworkException
	{
		assert (network != null);
		assert (trainingSet != null);
		assert (validationSet != null);
		assert (costFunction != null);
		
		logger.log(Level.INFO, "Training with stochastic gradient descent");
		for (epoch = 0; epoch < numEpochs; ++epoch) {	
			logger.log(Level.INFO, String.format("%s %d", "Starting epoch ", epoch));
			trainingSet.shuffle();			
			List<DataSet> batches = trainingSet.getMiniBatches(batchSize);
			batchNo = 0;
			// replaced the earlier for loop and moved the gradient calculation to a lambda
			// referencing a method of the class - this will ease separating the back-prop
			// aspect of the algorithm from the stochastic gradient descent
			batches.forEach(
					batch -> { 
							logger.log(Level.FINE, String.format("Training minibatch...%d", batchNo));
							batch.getDataInstances().forEach(
								(instance) -> { calculateAndCaptureGradient(network, instance, costFunction); }
											);
						
							
							adjustWeightsAndBiases(network);
							clearGradients();
							++batchNo;
						}
					);
			
			// TODO - early stopping on successful validation
			logger.log(Level.INFO, "Evaluating current epoch");
			Statistics stats = network.validateOrTest(validationSet, costFunction);
			stats.print(false);
			
		}	
	}
	

	private void feedForward(Network network, DataInstance instance) throws NetworkException
	{
		network.evaluate(instance);
	}
	
	private void backPropagate(Network network, DataInstance instance, CostFunction costFunction)
	{
		Layer outputLayer = network.getOutputLayer();
		
		
		// output layer error
		// outputLayer.setError(Vector.schurProduct(costFunction.costPrime(outputLayer.getActivation(), instance.getExpectedOutputs()),
		//	outputLayer.getActivationFunction().firstDerivative(outputLayer.getWeightedInput())));
		Vector costPrime = costFunction.costPrime(outputLayer.getActivation(), instance.getExpectedOutputs());
		Vector activation = outputLayer.getActivationFunction().firstDerivative(outputLayer.getWeightedInput());
		outputLayer.setError(costPrime.schurProduct(activation));
		Layer next = outputLayer;
		Layer prev = outputLayer.getPrevious();
		while (prev != null && prev.getActivationFunction() != null) {
			//prev.setError(Vector.schurProduct(Matrix.multiply(Matrix.transpose(next.getWeights()), next.getError()),
			//		prev.getActivationFunction().firstDerivative(prev.getWeightedInput())));
			
			prev.setError(
					next.getWeights().transpose().multiply(next.getError()).schurProduct(prev.getActivationFunction().firstDerivative(prev.getWeightedInput()))
			);
			
			next = prev;
			prev = next.getPrevious();
		}
	}
	
	private void captureGradient(Network network)
	{
		Layer layer = network.getOutputLayer();
		
		while (layer != null && layer.getPrevious() != null) {
			
			if (sumOfWeightGradients.containsKey(layer)) {
				Matrix newSum = sumOfWeightGradients.get(layer).add(layer.getWeightGradient());
				sumOfWeightGradients.put(layer, newSum);
			} else {
				sumOfWeightGradients.put(layer, layer.getWeightGradient());		
			}
			
			if (sumOfBiasGradients.containsKey(layer)) {
				Vector newSum = sumOfBiasGradients.get(layer).add(layer.getBiasGradient());
				sumOfBiasGradients.put(layer, newSum);
			} else {
				sumOfBiasGradients.put(layer, layer.getBiasGradient());
			}
			
			layer = layer.getPrevious();
		}
	}
	
	private void adjustWeightsAndBiases(Network network) 
	{
		try {
			Layer layer = network.getOutputLayer();
			while (layer != null && layer.getPrevious() != null) {
				
					adjustWeights(layer, learningRate, getMeanWeightGradient(layer));
				
				adjustBiases(layer, learningRate, getMeanBiasGradient(layer));
				layer = layer.getPrevious();
			}
		} catch (NetworkException e) {
			logger.log(Level.SEVERE, e.getMessage());
			throw new RuntimeException(e);
		}
	}
	
	private void clearGradients() 
	{
		sumOfWeightGradients.clear();
		sumOfBiasGradients.clear();
	}
	
	private Matrix getMeanWeightGradient(Layer layer)
	{
		return sumOfWeightGradients.get(layer).multiply(1.0/batchSize);
	}
	
	private Vector getMeanBiasGradient(Layer layer)
	{
		//return Vector.multiply(1.0/batchSize, sumOfBiasGradients.get(layer));
		return sumOfBiasGradients.get(layer).multiply(1.0/batchSize);
	}
	
	private void adjustWeights(Layer layer, double learningRate, Matrix meanWeightGradient) throws NetworkException {
		if (layer.getWeights() != null) {
			if (prevWeightDelta.containsKey(layer)) {
				//Matrix weightDelta = Matrix.add(
				//Matrix.multiply(-learningRate, meanWeightGradient),
				//		Matrix.multiply(momentum, prevWeightDelta.get(layer))
				//		);
				
				Matrix weightDelta = meanWeightGradient.multiply(-learningRate).add(prevWeightDelta.get(layer).multiply(momentum));
				layer.setWeights(layer.getWeights().add(weightDelta));
				prevWeightDelta.put(layer, weightDelta);
						
			} else {
		       layer.setWeights(layer.getWeights().subtract(meanWeightGradient.multiply(learningRate)));
		       prevWeightDelta.put(layer, meanWeightGradient.multiply(-learningRate));
			}
		}
	}

	private void adjustBiases(Layer layer, double learningRate, Vector meanBiasGradient) throws NetworkException {
		if (layer.getBiases() != null) {
			layer.setBiases(layer.getBiases().subtract(meanBiasGradient.multiply(learningRate)));
			//layer.setBiases(Vector.add(layer.getBiases(), Vector.multiply(-learningRate, meanBiasGradient)));
		}
	}
	
	public void calculateAndCaptureGradient(Network network, DataInstance instance, CostFunction costFunction)
	{
		try {
			//new BackPropagation().execute(instance, network, costFunction);
			feedForward(network, instance);
			backPropagate(network, instance, costFunction);
			captureGradient(network);
		} catch (NetworkException e) {
			logger.log(Level.SEVERE, e.getMessage());
			throw new RuntimeException(e);
		}
	}
	
}
