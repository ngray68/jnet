package jnet.net;

public interface CostFunction {

	/**
	 * Call this function to return the cost as a function of
	 * actual and expected output
	 * @param output
	 * @param expectedOutput
	 * @return
	 */
	public double cost(Vector output, Vector expectedOutput);
	
	/**
	 * Call this function to return the first derivative of the cost
	 * with respect to the activation (ie. output) value 
	 * @param output
	 * @param expectedOutput
	 * @return
	 */
	public Vector costPrime(Vector output, Vector expectedOutput);
	
}
