package jnet.algorithm;

import jnet.data.DataInstance;
import jnet.net.CostFunction;
import jnet.net.Layer;
import jnet.net.Matrix;
import jnet.net.Network;
import jnet.net.NetworkException;
import jnet.net.Vector;

/**
 * This class implements the back-propagation
 * algorithm for the given Network and CostFunction
 * for a single data instance
 * Calling method execute results in the network's
 * layers having weight and bias gradients computed
 * and available to other algorithms such as 
 * StochasticGradientDescent.
 * if the back-propagation fails an exception is thrown
 * @author nigelgray
 *
 */
public class BackPropagation {

	/**
	 * Executes the back-propagation algorithm on the given network for
	 * the specified instance and cost function
	 * Feed forward the instance's inputs to compute the network's output
	 * and then the cost function is useed to calculate the error which
	 * is back-propagated through the network to calculate the gradients
	 * of the cost function wrt to the weights and biases of the network.
	 * These can be obtained from the network after execution.
	 * An exception is thrown if the algorithm fails.
	 * @param instance
	 * @param network
	 * @param costFunction
	 * @throws NetworkException 
	 */
	public void execute(DataInstance instance, Network network, CostFunction costFunction) throws NetworkException
	{
		feedForward(network, instance);
		backPropagate(network, instance, costFunction);
	}
	
	private void feedForward(Network network, DataInstance instance) throws NetworkException
	{
		network.evaluate(instance);
	}
	
	private void backPropagate(Network network, DataInstance instance, CostFunction costFunction)
	{
		Layer outputLayer = network.getOutputLayer();
		
		// output layer error
		outputLayer.setError(Vector.schurProduct(costFunction.costPrime(outputLayer.getActivation(), instance.getExpectedOutputs()),
				outputLayer.getActivationFunction().firstDerivative(outputLayer.getWeightedInput())));
		
		Layer next = outputLayer;
		Layer prev = outputLayer.getPrevious();
		while (prev != null && prev.getActivationFunction() != null) {
			prev.setError(Vector.schurProduct(Matrix.multiply(Matrix.transpose(next.getWeights()), next.getError()),
					prev.getActivationFunction().firstDerivative(prev.getWeightedInput())));
			
			next = prev;
			prev = next.getPrevious();
		}
	}
}
