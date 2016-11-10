package jnet.net;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import jnet.data.DataInstance;
import jnet.data.DataSet;


public class Network {
	
	private List<Layer> layers;
	private static Logger logger = Logger.getLogger("Network");
	
	
	public Network(int[] layerSizes) {
		assert (layerSizes.length > 0);
		logger.log(Level.INFO, "Creating neural network\n");
		layers = new ArrayList<Layer>();
		layers.add(0, new Layer(layerSizes[0], null));
		for (int i = 1; i < layerSizes.length; ++i) {
			layers.add(i, new Layer(layerSizes[i], layers.get(i-1)));
		}
	}
	
	public void stochasticGradientDescent(DataSet dataSet, CostFunction costFunction, int numEpochs, int batchSize, double learningRate) {
		
		logger.log(Level.INFO, "Beginning training...");
		for (int epoch = 0; epoch < numEpochs; ++epoch) {
			logger.log(Level.INFO, String.format("%s %d", "Starting epoch ", epoch));
			dataSet.shuffle();			
			List<DataSet> batches = dataSet.getMiniBatches(batchSize);
			int count = 0;
			for (DataSet batch : batches) {
				if (count < batches.size() - 1) {	
					//logger.log(Level.INFO, String.format("Training minibatch %d", count));
					Iterator<DataInstance> instanceIter = batch.getIterator();
					while (instanceIter.hasNext()) {
						DataInstance instance = instanceIter.next();
						feedForward(instance);
						backPropagate(instance, costFunction);
						calculateGradient();
					}
					
					// adjust weights and biases
					for (Layer layer : layers) {
						layer.adjustWeights(learningRate);
						layer.adjustBiases(learningRate);
						layer.clearWeightGradients();
						layer.clearBiasGradients();
					}
				}
				else  {
					logger.log(Level.INFO, String.format("Evaluating minibatch %d", count));
					evaluate(batch);
				}
				++count;
			}	
		}
	}
	
	public void evaluate(DataSet dataSet) {
		Iterator<DataInstance> instanceIter = dataSet.getIterator();
		while (instanceIter.hasNext()) {
			DataInstance instance = instanceIter.next();
			Vector output = evaluate(instance);
			Vector expectedOutput = instance.getExpectedOutputs();
			Vector diff = Vector.add(expectedOutput, Vector.multiply(-1.0, output));
			double meansquarederror = 0.5 * Vector.dotProduct(diff, diff);
			logger.log(Level.INFO, String.format("Expected %s: Actual %s", expectedOutput.toString(), output.toString()));
			logger.log(Level.INFO, String.format("Difference %s MSE %f", diff.toString(), meansquarederror));
		}
	}
	
	public Vector evaluate(DataInstance instance) {
		feedForward(instance);
		return getOutput();
	}
	
	private Vector getOutput() {
		return layers.get(layers.size() - 1).getActivation();
	}
	
	
	
	public void feedForward(DataInstance dataInstance) {
		assert (layers.size() > 0);
		layers.get(0).setActivation(dataInstance.getInputs());
		for (Layer layer : layers) {
			layer.feedForward();
		}	
	}
	
	public void backPropagate(DataInstance dataInstance, CostFunction costFunction) {
		Layer outputLayer = layers.get(layers.size() - 1);
		outputLayer.backPropagate(dataInstance, costFunction);
		Layer next = outputLayer;
		Layer prev = outputLayer.getPrevious();
		while (prev != null) {
			prev.backPropagate(next);
			next = prev;
			prev = next.getPrevious();
		}
	}
	
	public void calculateGradient() {
		Layer layer = layers.get(layers.size() - 1);
		
		while (layer != null && layer.getPrevious() != null) {
			layer.addBiasGradient(layer.getBiasGradient());
			layer.addWeightGradient(layer.getWeightGradient());
			layer = layer.getPrevious();
		}
	}
	
	// TODO Refactor project
	// TODO Check into Github
	// TODO Refactor Network to separate Learning algorithm
	// TODO Review interfaces, implementation
	// TODO Formalize statistics gathering
	// TODO Formalize data set reading
	// TODO add logging to Network and Layer classes IN PROGRESS
	// TODO complete unit tests IN PROGRESS
	// TODO migrate to Maven build
}