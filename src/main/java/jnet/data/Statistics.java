package jnet.data;

import java.util.HashMap;
import java.util.Map;

import jnet.net.CostFunction;
import jnet.net.Vector;

/**
 * Class Statistics
 * This class gathers performance statistics for a neural network
 * @author nigelgray
 *
 */
public class Statistics {
	
	//private int epoch;
	//private DataSet dataSet;
	private Map<DataInstance, Double> errors;
	private Map<DataInstance, Vector> outputs;
	private double sumError;
	private ConfusionMatrix confusionMatrix;
	
	/**
	 * Construct an empty Statistics object
	 */
	public Statistics()
	{
	}
	
	/**
	 * Update the statistics for the given data instance, output and cost function
	 * @param instance
	 * @param networkOutput
	 * @param costFunction
	 */
	public void addStatistics(DataInstance instance, Vector networkOutput, CostFunction costFunction)
	{
		if (errors == null)
			errors = new HashMap<>();
		if (outputs == null)
			outputs = new HashMap<>();
		
		
		double error = costFunction.cost(networkOutput, instance.getExpectedOutputs());
		errors.put(instance, error);
		sumError += error;
	
		outputs.put(instance, networkOutput);
		
		if (confusionMatrix == null) {
			confusionMatrix = new ConfusionMatrix(instance.getExpectedOutputs(), networkOutput);
		} else {
			confusionMatrix.addDataInstance(instance.getExpectedOutputs(), networkOutput);
		}
	}
	
	/**
	 * Get the mean error across all the instances gathered by this Statistics object
	 * @return
	 */
	public double getMeanError()
	{
		assert (errors.size() > 0);
		return sumError/errors.size();
	}
	
	/**
	 * Print the information stored in this statistics object
	 * @param detailed
	 */
	public void print(boolean detailed)
	{	
		if (detailed) {		
			outputs.forEach(
				(instance, output) -> {
						System.out.println("Expected output:" + instance.getExpectedOutputs().toString());
						System.out.println("Actual output:" + output.toString());
						System.out.println("Error: " + errors.get(instance).toString());
						System.out.println();
					}
				);
		}
		
		System.out.println(String.format("Mean error: %f", getMeanError()));
		System.out.println(String.format("Accuracy: %f", getAccuracy()));
		System.out.println(String.format("Precision: %s", getPrecision().toString()));
		System.out.println(String.format("Recall: %s\n", getRecall().toString()));
	
		
	}
 
	/**
	 * Return the accuracy
	 * @return
	 */
	public double getAccuracy()
	{
		assert (confusionMatrix != null);
		return confusionMatrix.getAccuracy();
	}

	/**
	 * Return the precision
	 * @return
	 */
	public Vector getPrecision()
	{
		assert (confusionMatrix != null);
		return confusionMatrix.getPrecision();
	}
	
	/**
	 * Return the accuracy
	 * @return
	 */
	public Vector getRecall()
	{
		assert (confusionMatrix != null);
		return confusionMatrix.getRecall();
	}
}
