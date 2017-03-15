package jnet.net;

import com.ngray.jnet.algebra.Vector;

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
	 * Call this method to train the neural network with the given trainingSet, where the
	 * testSet is separately supplied
	 * @param trainingSet
	 * @param validationSet
	 * @param algorithm
	 * @param costFunction
	 * @throws NetworkException
	 */
	void train(DataSet trainingSet, DataSet testSet, LearningAlgorithm algorithm, CostFunction costFunction) throws NetworkException;
		
	/**
	 * Call this method to validate or test a trained or partially trained 
	 * neural network
	 * @param dataSet
	 * @param costFunction
	 * @return statistics measuring the validation results
	 * @throws NetworkException 
	 */
	public Statistics validateOrTest(DataSet dataSet, CostFunction costFunction) throws NetworkException;
	
	/**
	 * Call this method to apply the trained or partially trained neural network
	 * to the chosen data instance
	 * @param instance
	 * @return the output of the neural network for the given data instance
	 * @throws NetworkException 
	 */
	public Vector evaluate(DataInstance instance) throws NetworkException;
	
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
