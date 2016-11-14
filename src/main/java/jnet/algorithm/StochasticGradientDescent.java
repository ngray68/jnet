package jnet.algorithm;

import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.data.Statistics;
import jnet.net.CostFunction;
import jnet.net.Layer;
import jnet.net.Network;

public class StochasticGradientDescent implements LearningAlgorithm {

	private static Logger logger = Logger.getGlobal();
	
	private int numEpochs;
	private int batchSize;
	private double learningRate;
	//private double momentum;
	
	public StochasticGradientDescent(int numEpochs, int batchSize, double learningRate, double momentum) {
		this.numEpochs = numEpochs;
		this.batchSize = batchSize;
		this.learningRate = learningRate;
		//this.momentum = momentum;
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
		
			int count = 1;
			for (DataSet batch : batches) {		
				logger.log(Level.FINE, String.format("Training minibatch %d", count));
				Iterator<DataInstance> instanceIter = batch.getIterator();
				while (instanceIter.hasNext()) {
					DataInstance instance = instanceIter.next();
					feedForward(network, instance);
					backPropagate(network, instance, costFunction);
					calculateGradient(network);
				}
				++count;
			
				adjustWeightsAndBiases(network);
			}
			
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
		outputLayer.backPropagate(instance, costFunction);
		Layer next = outputLayer;
		Layer prev = outputLayer.getPrevious();
		while (prev != null) {
			prev.backPropagate(next);
			next = prev;
			prev = next.getPrevious();
		}
	}
	
	private void calculateGradient(Network network)
	{
		Layer layer = network.getOutputLayer();
		
		while (layer != null && layer.getPrevious() != null) {
			layer.addBiasGradient(layer.getBiasGradient());
			layer.addWeightGradient(layer.getWeightGradient());
			layer = layer.getPrevious();
		}
	}
	
	private void adjustWeightsAndBiases(Network network) 
	{
		// adjust weights and biases
		Layer layer = network.getOutputLayer();
		while (layer != null) {
			layer.adjustWeights(learningRate);
			layer.adjustBiases(learningRate);
			layer.clearWeightGradients();
			layer.clearBiasGradients();
			layer = layer.getPrevious();
		}
	}
}
