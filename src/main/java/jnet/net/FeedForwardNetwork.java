package jnet.net;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import jnet.algorithm.LearningAlgorithm;
import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.data.Statistics; 


public class FeedForwardNetwork implements Network {
	
	private List<Layer> layers;
	private static Logger logger = Logger.getGlobal();
	
	public FeedForwardNetwork(int[] layerSizes) 
	{
		assert (layerSizes.length > 0);
		logger.log(Level.INFO, "Creating neural network\n");
		layers = new ArrayList<Layer>();
		layers.add(0, new Layer(layerSizes[0], null, null));
		for (int i = 1; i < layerSizes.length; ++i) {
			layers.add(i, new Layer(layerSizes[i], layers.get(i-1), new SigmoidFunction()));
		}
	}
	/*
	@Override
	public void train(DataSet trainingSet, DataSet testSet, LearningAlgorithm algorithm, CostFunction costFunction) throws NetworkException
	{
		logger.log(Level.INFO, "Starting training...");
		if (trainingSet == null || testSet == null)
			throw new NetworkException("DataSet cannot be null");
		if (algorithm == null)
			throw new NetworkException("LearningAlgorithm cannot be null");
		if (costFunction == null)
			throw new NetworkException("CostFunction cannot be null");
		
		if (trainingSet.isEmpty() || testSet.isEmpty()) {
			logger.log(Level.WARNING, "Network.train() called with empty dataset - doing nothing");
			return;
		}
		
		// make sure the data set is shuffled
		trainingSet.shuffle();
		
		// split the data set into training, validation sets
		algorithm.execute(this, trainingSet.getTrainingSubset(), trainingSet.getValidationSubset(), costFunction);

		Statistics testStats = validateOrTest(testSet, costFunction);
		testStats.print(true);
		
	}*/
	
	@Override
	public void train(DataSet trainingSet, DataSet validationSet, LearningAlgorithm algorithm, CostFunction costFunction) throws NetworkException 
	{
		logger.log(Level.INFO, "Starting training...");
		if (trainingSet == null)
			throw new NetworkException("Training DataSet cannot be null");
		if (validationSet == null)
			throw new NetworkException("Validation DataSet cannot be null");
		if (algorithm == null)
			throw new NetworkException("LearningAlgorithm cannot be null");
		if (costFunction == null)
			throw new NetworkException("CostFunction cannot be null");
		
		if (trainingSet.isEmpty()) {
			throw new NetworkException("Training DataSet empty");
		}
		
		if (validationSet.isEmpty()) {
			throw new NetworkException("Validation DataSet empty");
		}
		
		// make sure the data set is shuffled
		trainingSet.shuffle();
		algorithm.execute(this, trainingSet, validationSet, costFunction);
	}

	@Override
	public Statistics validateOrTest(DataSet dataSet, CostFunction costFunction) throws NetworkException 
	{
		assert (dataSet != null);
		Statistics stats = new Statistics();
		
		for(Iterator<DataInstance> instanceIter = dataSet.getIterator(); instanceIter.hasNext(); ) {
			DataInstance instance = instanceIter.next();
			if (instance == null) {
				throw new NetworkException("Cannot evualate null data instance");
			}
			stats.addStatistics(instance, evaluate(instance), costFunction);
		}
		return stats;
	}
	
	@Override
	public Vector evaluate(DataInstance instance) throws NetworkException 
	{
		for (Layer layer : layers) {
			if (layer == null) {
				throw new NetworkException("Null pointer to network Layer detected - stopping evaluation");
			}
			if (layer.getPrevious() == null || layer.getActivationFunction() == null) {
				layer.setActivation(instance.getInputs());
			}
			else {
				layer.setWeightedInput(layer.getPrevious().getActivation());
				layer.setActivation(layer.getActivationFunction().evaluate(layer.getWeightedInput()));
			}
		}	
		return getOutput();
	}
		
	@Override
	public Layer getInputLayer()
	{
		//checkIndex(0);
		return layers.get(0);
	}
	
	@Override
	public Layer getOutputLayer()
	{
		//checkIndex(layers.size() - 1));
		return layers.get(layers.size() - 1);
	}
	
	
	private Vector getOutput() {
		return getOutputLayer().getActivation();
	}
	
	// TODO Add defensive checks to all classes and test these cases
	// TODO complete unit tests IN PROGRESS
	
}