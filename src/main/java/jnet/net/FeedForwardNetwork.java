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
		layers.add(0, new Layer(layerSizes[0], null));
		for (int i = 1; i < layerSizes.length; ++i) {
			layers.add(i, new Layer(layerSizes[i], layers.get(i-1)));
		}
	}
	
	@Override
	public void train(DataSet dataSet, LearningAlgorithm algorithm, CostFunction costFunction) 
	{
		logger.log(Level.INFO, "Starting training...");
		if (dataSet == null)
			throw new NullPointerException("DataSet cannot be null");
		if (algorithm == null)
			throw new NullPointerException("LearningAlgorithm cannot be null");
		if (costFunction == null)
			throw new NullPointerException("CostFunction cannot be null");
		
		if (dataSet.isEmpty()) {
			logger.log(Level.WARNING, "Network.train() called with empty dataset - doing nothing");
			return;
		}
		
		
		// split the data set into training, validation and test sets
		DataSet trainingSet = dataSet.getTrainingSubset();
		DataSet validationSet = dataSet.getValidationSubset();
		DataSet testSet = dataSet.getTestSubset();
		
		algorithm.execute(this, trainingSet, validationSet, costFunction);

		Statistics testStats = validateOrTest(testSet, costFunction);
		testStats.print(true);
	}

	@Override
	public Statistics validateOrTest(DataSet dataSet, CostFunction costFunction) 
	{
		assert (dataSet != null);
		Statistics stats = new Statistics();
		Iterator<DataInstance> instanceIter = dataSet.getIterator();
		while (instanceIter.hasNext()) {
			
			DataInstance instance = instanceIter.next();
			stats.addStatistics(instance, evaluate(instance), costFunction);
		}
		return stats;
	}
	
	@Override
	public Vector evaluate(DataInstance instance) 
	{
		layers.get(0).setActivation(instance.getInputs());
		for (Layer layer : layers) {
			layer.feedForward();
		}	
		return getOutput();
	}
	
	@Override
	public void adjustWeightsAndBiases(List<Matrix> deltaWeight, List<Vector> deltaBias) 
	{
		// TODO	
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
	
	
	// TODO Check into Github DONE
	// TODO migrate to Maven build  DONE
	// TODO Refactor Network to separate Learning algorithm IN PROGRESS
	// TODO Review interfaces, implementation IN PROGRESS
	// TODO Formalize statistics gathering IN PROGRESS
	// TODO Formalize data set reading
	// TODO Add defensive checks to all classes and test these cases
	// TODO add logging to Network and Layer classes IN PROGRESS
	// TODO complete unit tests IN PROGRESS
	
}