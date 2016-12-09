package jnet.net;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public interface CostFunction {

	final static Map<String, CostFunction> functions = new HashMap<>();
	
	/**
	 * Factory method for cost functions
	 * @param costFunctionType
	 * @return
	 * @throws NetworkException
	 */
	public static CostFunction create(String costFunctionType) throws NetworkException
	{
		if (functions.isEmpty()) {
			functions.put("Quadratic", new QuadraticCostFunction());
			functions.put("CrossEntropy", new CrossEntropyCostFunction());
		}
		if (!functions.containsKey(costFunctionType)) {
			throw new NetworkException("Unknown cost function type " + costFunctionType);
		}
		return functions.get(costFunctionType);
	}
	
	/**
	 * Get a collection of supported cost function types
	 * @return
	 */
	public static Set<String> getCostFunctionTypes()
	{
		if (functions.isEmpty()) {
			functions.put("Quadratic", new QuadraticCostFunction());
			functions.put("CrossEntropy", new CrossEntropyCostFunction());
		}
		return functions.keySet();
	}
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
