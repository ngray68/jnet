package jnet.algorithm;

import jnet.data.DataSet;
import jnet.net.CostFunction;
import jnet.net.Network;
import jnet.net.NetworkException;

/**
 * Interface to be implemented by LearningAlgorithms
 * @author nigelgray
 *
 */
public interface LearningAlgorithm {
	
	/**
	 * Execute the implemented learning algorithm. Throw if the algorithm fails
	 * @param network
	 * @param trainingSet
	 * @param validationSet
	 * @param costFunction
	 * @throws NetworkException
	 */
	public void execute(Network network, DataSet trainingSet, DataSet validationSet, CostFunction costFunction) throws NetworkException;
}
