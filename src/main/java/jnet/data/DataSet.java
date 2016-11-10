package jnet.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
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
	
	// Create a new empty data set
	public DataSet() {
		dataInstances = new TreeMap<Integer, DataInstance>();
	}
	
	// Create a data set from the sorted map
	// Ownership of the sorted map does not pass to this data set object
	// Private constructor used only in getMiniBatches()
	private DataSet(SortedMap<Integer, DataInstance> dataSet) {
		this.dataInstances = dataSet;
	}
	
	public void addInstance(DataInstance instance) {
		if (dataInstances.isEmpty())
		{
			dataInstances.put(0, instance);
			return;
		}
			
		dataInstances.put(dataInstances.lastKey() + 1, instance);
	}
	
    public Iterator<DataInstance> getIterator() {
		return dataInstances.values().iterator();
	}
    
    public void normalize() {
    	
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
	
	public void shuffle() {
		TreeMap<Integer, DataInstance> shuffledInstances = new TreeMap<Integer, DataInstance>();
		
		Iterator<DataInstance> iter = getIterator();
		DataInstance instance = null;
		while (instance != null || iter.hasNext()) {
			if (instance == null)
			   instance = iter.next();
			int randomIndex = ThreadLocalRandom.current().nextInt(0, getNumInstances());
		
			if (!shuffledInstances.containsKey(randomIndex)) {
				shuffledInstances.put(randomIndex, instance);
				instance = null;
			}
		}
		dataInstances = shuffledInstances;
	}
	
	public List<DataSet> getMiniBatches(int miniBatchSize) {
		ArrayList<DataSet> miniBatches = new ArrayList<DataSet>();
				
		for (int i = 0; i < dataInstances.size() - miniBatchSize; i= i + miniBatchSize) {
			DataSet newMiniBatch = new DataSet(dataInstances.subMap(i, i + miniBatchSize));
			miniBatches.add(newMiniBatch);
		}
		return miniBatches;
	}
	
	public int getNumInstances() {
		return dataInstances.size();
	}
	
	public void readFromFile(String fileName) {
		
		BufferedReader buffer = null;
		try {
			FileReader reader = new FileReader(fileName);
			buffer = new BufferedReader(reader);
			String line;
			while ((line = buffer.readLine()) != null) {
				// for now assume each line as 1 expected output
				// and the rest are the inputs
				String[] entries = line.split(",");
				ArrayList<Double> expectedOutputs = new ArrayList<Double>();
				if (entries[0].equals("1")) {
					expectedOutputs.add(1.0);
					expectedOutputs.add(0.0);
					expectedOutputs.add(0.0);	
				}
				else if (entries[0].equals("2")) {
					expectedOutputs.add(0.0);
					expectedOutputs.add(1.0);
					expectedOutputs.add(0.0);	
				}
				else if (entries[0].equals("3")) {
					expectedOutputs.add(0.0);
					expectedOutputs.add(0.0);
					expectedOutputs.add(1.0);
				}
				
				ArrayList<Double> inputs = new ArrayList<Double>();
				for (int i = 1; i < entries.length; ++i) {
					inputs.add(Double.parseDouble(entries[i]));
				}
				DataInstance instance = new DataInstance(inputs, expectedOutputs);
				addInstance(instance);
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			//if (buffer != null)
			
		}
	}
}
