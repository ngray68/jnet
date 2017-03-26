package com.ngray.jnet.recurrent;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.ngray.jnet.algebra.Vector;

public final class SequenceGenerator<E> {

	private final Dictionary<E> dictionary;
	
	/**
	 * Construct a sequence generator object from the dictionary
	 * @param dictionary
	 */
	public SequenceGenerator(Dictionary<E> dictionary) {
		this.dictionary = dictionary.copy();
	}
	
	/**
	 * Convert the list of E into its equivalent
	 * sequence using the dictionary owned by this
	 * class
	 * @param rawData
	 * @return
	 * @throws SequenceException 
	 */
	public Sequence getInputSequence(List<E> rawData) throws SequenceException {
		List<Vector> vectors = new ArrayList<>();
		rawData.forEach(e -> vectors.add(dictionary.getVector(e)));
		return Sequence.newSequence(vectors);
	}
	
	/**
	 * Generate the Sequence with which we label the raw data
	 * input. This is the expected output from the RNN which we
	 * use to calculate the loss function by comparison with the
	 * actual output generated
	 * @param rawData
	 * @return
	 * @throws SequenceException 
	 */
	public Sequence getOutputSequence(List<E> rawData) throws SequenceException {
		List<Vector> vectors = new ArrayList<>();
		if (rawData.size() == 0)
			return Sequence.newSequence(vectors);
		
		rawData.subList(1, rawData.size()).forEach(e -> vectors.add(dictionary.getVector(e)));
		
		// TODO replace with end_seq token
		vectors.add(dictionary.getVector(rawData.get(0)));
		return Sequence.newSequence(vectors);
	}
	
	
	
	public List<E> convertSequence(Sequence sequence) {
		List<E> elements = new ArrayList<>();
		Vector v = null;
		Iterator<Vector> iter = sequence.getIterator();
		while (iter.hasNext()) {
			v = iter.next();
			elements.add(dictionary.getInstance(v));
		}	
		return elements;
	}
	
	
}
