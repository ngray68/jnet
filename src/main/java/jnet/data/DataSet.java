package jnet.data;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import jnet.net.Vector;

public class DataSet {

	private List<DataInstance> dataInstances;
	
	private double trainingSetFraction;
	private double validationSetFraction;
	private double testSetFraction;
	
	
/* Public constructors */
	
/* Public static methods */

	public static DataSet create(double trainingSetFraction) throws DataException 
	{
		if (trainingSetFraction > 1.0)
			throw new DataException("DataSet training subset cannot be larger than the parent dataset (choose Training Set Fraction <= 1.0)");
		return new DataSet(trainingSetFraction);
	}
	
	public static DataSet create(double trainingFraction, double validationFraction) throws DataException {
		if (trainingFraction + validationFraction > 1.0)
			throw new DataException("DataSet training subset cannot be larger than the parent dataset (choose Training Set Fraction + Validation Set Fraction <= 1.0)");
		return new DataSet(trainingFraction, validationFraction);
	}

	
	public static DataSet create()
	{
		return new DataSet();
	}
	
/* Public instance methods */
	
	public void normalize() 
	{
    	Iterator<DataInstance> iter = getIterator();
    	double[] maxValues = null;
    	double[] minValues = null;
    	while (iter.hasNext()) {
    		Vector inputs = iter.next().getInputs();
    		if (maxValues == null)
    			maxValues = new double[inputs.getSize()];
    		if (minValues == null)
    			minValues = new double[inputs.getSize()];
    		for (int i = 0; i < inputs.getSize(); ++i) {
    			minValues[i] =  minValues[i] > inputs.getElement(i) ? inputs.getElement(i) : minValues[i];
    			maxValues[i] =  maxValues[i] < inputs.getElement(i) ? inputs.getElement(i) : maxValues[i];
    		}
    	}
    	
    	iter = getIterator();
    	while (iter.hasNext()) {
    		DataInstance instance = iter.next();
    		instance.normalize(minValues, maxValues);
    	}
    }
	
	/**
	 * Method shuffle()
	 * Randomly shuffles the data instances in the set
	 */
	public void shuffle() 
	{
		Collections.shuffle(dataInstances);
	}
	
	/**
	 * Method  getTrainingSubset
	 * @return a DataSet containing the sublist of training instances
	 *         if the parent set is empty, returns the same empty DataSet
	 */
	public DataSet getTrainingSubset() 
	{
		if (dataInstances.isEmpty())
			return this;
		
		return new DataSet(dataInstances.subList(0, getNumTrainingInstances()));	
	}
	
	/**
	 * Method  getValidationSubset
	 * @return a DataSet containing the sublist of validation instances
	 *         if the parent set is empty, returns the same empty DataSet
	 */
	public DataSet getValidationSubset()
	{
		if (dataInstances.isEmpty())
			return this;
		
		int startingIndex = getNumTrainingInstances();
		return new DataSet(dataInstances.subList(startingIndex, startingIndex + getNumValidationInstances()));
	}
	
	/**
	 * Method  getTestSubset
	 * @return a DataSet containing the sublist of test instances
	 *         if the parent set is empty, returns the same empty DataSet
	 */
	public DataSet getTestSubset()
	{
		if (dataInstances.isEmpty())
			return this;
		
		int startingIndex = getNumTrainingInstances() + getNumValidationInstances();
		return new DataSet(dataInstances.subList(startingIndex, startingIndex + getNumTestInstances()));
	}
	
	
	
	public List<DataSet> getMiniBatches(int miniBatchSize) 
	{
		ArrayList<DataSet> miniBatches = new ArrayList<DataSet>();
				
		for (int i = 0; i < dataInstances.size() - miniBatchSize; i= i + miniBatchSize) {
			DataSet newMiniBatch = new DataSet(dataInstances.subList(i, i + miniBatchSize));
			miniBatches.add(newMiniBatch);
		}
		return miniBatches;
	}
	
	public void addInstance(DataInstance instance) 
	{		
		dataInstances.add(instance);
	}
	
    public Iterator<DataInstance> getIterator() 
    {
		return dataInstances.iterator();
	}
    
    public List<DataInstance> getDataInstances()
    {
    	return dataInstances;
    }
    
	public int getNumInstances() 
	{
		return dataInstances.size();
	}
	
	public boolean isEmpty() 
	{
		return dataInstances.isEmpty();
	}

/* Private constructors */

	// Create a new empty data set
	// with default training, validation and test set fractions
	private DataSet() {
		dataInstances = new ArrayList<>();
		trainingSetFraction = 0.8;
		validationSetFraction = 0.1;
		testSetFraction = 0.1;
	}
	
	private DataSet(double trainingSetFraction)
	{
		this.dataInstances = new ArrayList<>();
		this.trainingSetFraction = trainingSetFraction;
		this.validationSetFraction = 0.5 * (1.0 - trainingSetFraction);
		this.testSetFraction = validationSetFraction;
	}
	
	private DataSet(double trainingSetFraction, double validationSetFraction)
	{
		this.dataInstances = new ArrayList<>();
		this.trainingSetFraction = trainingSetFraction;
		this.validationSetFraction = validationSetFraction;
		this.testSetFraction = 1.0 - trainingSetFraction - validationSetFraction;
	}
	
	private DataSet(List<DataInstance> dataSet) 
	{
		this.dataInstances = dataSet;
		this.trainingSetFraction = 1.0;
		this.validationSetFraction = 0.0;
		this.testSetFraction = 0.0;
	}

/* Private helper methods */
	
	private int getNumTrainingInstances()
	{
		return (int)(dataInstances.size() * trainingSetFraction);
	}
	
	private int getNumValidationInstances()
	{
		return (int)(dataInstances.size() * validationSetFraction);
	}
	
	private int getNumTestInstances()
	{
		return (int)(dataInstances.size() * testSetFraction);
	}	
}
