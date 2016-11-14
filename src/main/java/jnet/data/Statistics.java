package jnet.data;

import java.util.HashMap;
import java.util.Map;

import jnet.net.CostFunction;
import jnet.net.Vector;

public class Statistics {
	
	//private int epoch;
	//private DataSet dataSet;
	private Map<DataInstance, Double> errors;
	private Map<DataInstance, Vector> outputs;
	private double sumError;
	//private ConfusionMatrix confusionMatrix;
	
	
	public Statistics()
	{
		
	}
	
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
	}
	
	public double getMeanError()
	{
		assert (errors.size() > 0);
		return sumError/errors.size();
	}
	
	public void print(boolean detailed)
	{
		System.out.println(String.format("Mean error: %f", getMeanError()));
	
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
		
	}
/*	
 
 	public void addToConfusionMatrix(DataInstance instance, Vector networkOutput) 
	{
		
	}
	public double getAccuracy()
	{
	}

	public double getPrecision()
	{
	}
	
	public double getRecall()
	{
		
	}
	*/
}
