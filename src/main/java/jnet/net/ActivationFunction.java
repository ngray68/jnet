package jnet.net;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.ngray.jnet.algebra.Vector;

/**
 * Network activation functions implement this interface
 * Note that despite the name, this is not a functional interface
 * @author nigelgray
 *
 */
public interface ActivationFunction {
	
	final static Map<String, ActivationFunction> functions = new HashMap<>();
	
	/**
	 * Factory method for activation functions
	 * @param activationFunctionType
	 * @return
	 * @throws NetworkException
	 */
	public static ActivationFunction create(String activationFunctionType) throws NetworkException
	{
		if (functions.isEmpty()) {
			functions.put("Sigmoid", new SigmoidFunction());
		}
		if (!functions.containsKey(activationFunctionType)) {
			throw new NetworkException("Unknown activation function type " + activationFunctionType);
		}
		return functions.get(activationFunctionType);
	}
	
	/**
	 * Get a collection of supported activation function types
	 * @return
	 */
	public static Set<String> getFunctionTypes()
	{
		if (functions.isEmpty()) {
			functions.put("Sigmoid", new SigmoidFunction());
		}
		return functions.keySet();
	}
	
	/**
	 * Implement this function as a function of the weightedInput
	 * eg. sigmoid function 1/(1-exp(-weightedInput))
	 * @param weightedInput
	 * @return
	 */
	public double evaluate(double weightedInput);
	
	/**
	 * Implement this function as the first derivative wrt to
	 * the weightedInput
	 * @param weightedInput
	 * @return
	 */
	public double firstDerivative(double weightedInput);
	
	/**
	 * Vectorized version of function(double)
	 * @param weightedInputs
	 * @return
	 */
	public Vector evaluate(Vector weightedInputs);
	
	/**
	 * Vectorized version of firstDerivative(double)
	 * @param weightedInputs
	 * @return
	 */
	public Vector firstDerivative(Vector weightedInputs);
}
