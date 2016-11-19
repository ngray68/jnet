package jnet.data;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ThreadLocalRandom;

import jnet.net.Vector;

public class DataSet {

	// store the dataInstances in a SortedMap rather than an array
	// or more basic collection
	// this allows us to find subsets for each mini-batch easily and
	// efficiently
	private SortedMap<Integer, DataInstance> dataInstances;
	
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
	
	public static DataSet create()
	{
		return new DataSet();
	}
	
/* Public instance methods */
	
	public void normalize() 
	{
    	Iterator<DataInstance> iter = getIterator();
    	Double[] maxValues = null;
    	Double[] minValues = null;
    	while (iter.hasNext()) {
    		Vector inputs = iter.next().getInputs();
    		if (maxValues == null)
    			maxValues = new Double[inputs.getSize()];
    		if (minValues == null)
    			minValues = new Double[inputs.getSize()];
    		for (int i = 0; i < inputs.getSize(); ++i) {
    			minValues[i] = minValues[i] == null || minValues[i] > inputs.getElement(i) ? inputs.getElement(i) : minValues[i];
    			maxValues[i] = maxValues[i] == null || maxValues[i] < inputs.getElement(i) ? inputs.getElement(i) : maxValues[i];
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
	 * Assumes that keys are assigned contiguously in all data sets
	 */
	public void shuffle() 
	{
		TreeMap<Integer, DataInstance> shuffledInstances = new TreeMap<Integer, DataInstance>();
		
		Iterator<DataInstance> iter = getIterator();
		int minKey = dataInstances.firstKey();
		int maxKey = dataInstances.lastKey();
		DataInstance instance = null;
		while (instance != null || iter.hasNext()) {
			if (instance == null)
			   instance = iter.next();
			int randomIndex = ThreadLocalRandom.current().nextInt(minKey, maxKey + 1);
		
			if (!shuffledInstances.containsKey(randomIndex)) {
				shuffledInstances.put(randomIndex, instance);
				instance = null;
			}
		}
		dataInstances = shuffledInstances;
	}
	
	/**
	 * Method  getTrainingSubset
	 * @return a DataSet containing the submap of training instances
	 *         if the parent set is empty, returns the same empty DataSet
	 */
	public DataSet getTrainingSubset() 
	{
		if (dataInstances.isEmpty())
			return this;
		
		return new DataSet(dataInstances.subMap(0, getNumTrainingInstances()));	
	}
	
	/**
	 * Method  getValidationSubset
	 * @return a DataSet containing the submap of validation instances
	 *         if the parent set is empty, returns the same empty DataSet
	 */
	public DataSet getValidationSubset()
	{
		if (dataInstances.isEmpty())
			return this;
		
		int startingIndex = getNumTrainingInstances();
		return new DataSet(dataInstances.subMap(startingIndex, startingIndex + getNumValidationInstances()));
	}
	
	/**
	 * Method  getTestSubset
	 * @return a DataSet containing the submap of test instances
	 *         if the parent set is empty, returns the same empty DataSet
	 */
	public DataSet getTestSubset()
	{
		if (dataInstances.isEmpty())
			return this;
		
		int startingIndex = getNumTrainingInstances() + getNumValidationInstances();
		return new DataSet(dataInstances.subMap(startingIndex, startingIndex + getNumTestInstances()));
	}
	
	
	
	public List<DataSet> getMiniBatches(int miniBatchSize) 
	{
		ArrayList<DataSet> miniBatches = new ArrayList<DataSet>();
				
		for (int i = 0; i < dataInstances.size() - miniBatchSize; i= i + miniBatchSize) {
			DataSet newMiniBatch = new DataSet(dataInstances.subMap(i, i + miniBatchSize));
			miniBatches.add(newMiniBatch);
		}
		return miniBatches;
	}
	
	public void addInstance(DataInstance instance) 
	{
		if (dataInstances.isEmpty())
		{
			dataInstances.put(0, instance);
			return;
		}
			
		dataInstances.put(dataInstances.lastKey() + 1, instance);
	}
	
    public Iterator<DataInstance> getIterator() 
    {
		return dataInstances.values().iterator();
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
		dataInstances = new TreeMap<Integer, DataInstance>();
		trainingSetFraction = 0.8;
		validationSetFraction = 0.1;
		testSetFraction = 0.1;
	}
	
	private DataSet(double trainingSetFraction)
	{
		this.dataInstances = new TreeMap<Integer, DataInstance>();
		this.trainingSetFraction = trainingSetFraction;
		this.validationSetFraction = 0.5 * (1.0 - trainingSetFraction);
		this.testSetFraction = validationSetFraction;
	}
	
	private DataSet(SortedMap<Integer, DataInstance> dataSet) 
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
