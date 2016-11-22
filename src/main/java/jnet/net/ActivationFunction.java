package jnet.net;

/**
 * Network activation functions implement this interface
 * @author nigelgray
 *
 */
public interface ActivationFunction {
	
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
