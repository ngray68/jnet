package jnet.net;

import java.util.List;

import jnet.algorithm.LearningAlgorithm;
import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.data.Statistics;

/**
 * Interface for all neural network implementations
 * @author nigelgray
 *
 */
public interface Network {

	/**
	 * Call this method to train the neural network with the given
	 * data set, learning algorithm, and cost function
	 * Hyper-parameters are provided by the learning algorithm
	 * @param trainingSet
	 * @param algorithm
	 * @param costFunction
	 */
	public void train(DataSet dataSet, LearningAlgorithm algorithm, CostFunction costFunction);
	
	/**
	 * Call this method to validate or test a trained or partially trained 
	 * neural network
	 * @param dataSet
	 * @param costFunction
	 * @return statistics measuring the validation results
	 */
	public Statistics validateOrTest(DataSet dataSet, CostFunction costFunction);
	
	/**
	 * Call this method to apply the trained or partially trained neural network
	 * to the chosen data instance
	 * @param instance
	 * @return the output of the neural network for the given data instance
	 */
	public Vector evaluate(DataInstance instance);
	
	/**
	 * Call this method to return a reference to the network's input layer
	 * @return Layer
	 */
	public Layer getInputLayer();
	
	/**
	 * Call this method to return a reference to the network's output layer
	 * @return Layer
	 */
	public Layer getOutputLayer();
}
