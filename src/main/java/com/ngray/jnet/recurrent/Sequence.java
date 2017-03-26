package com.ngray.jnet.recurrent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


import com.ngray.jnet.algebra.Vector;

/**
 * Class Sequence represents the fundamental data type that 
 * RNNs (recurrent neural nets) operate on. The sequence is
 * essentially an ordered list of Vectors (of doubles), each vector
 * representing the input at a given time-step to the network
 * @author nigelgray
 *
 */
public final class Sequence {

	/**
	 * The underlying sequence of elements. Each vector must be
	 * of equal size
	 */
	private final List<Vector>  elements;
	
	private Sequence(List<Vector> elements) {
		this.elements = new ArrayList<>(elements);
	}
	
	private static void checkElements(List<Vector> elements) throws SequenceException {
		int size = -1;
		for (Iterator<Vector> iter = elements.iterator(); iter.hasNext(); ) {
			if(size == -1) {
				size = iter.next().getSize();
				continue;
			}
			if (size != iter.next().getSize()) {
					throw new SequenceException("Invalid sequence - detected differently sized vectors");
			}
		}	
	}
	
	/**
	 * Static factory method to return a new sequence.
	 * Throws if the sequence elements are not the same size. Note that an empty sequence is
	 * valid.
	 * This is the only way to create a new sequence 
	 * @param elements
	 * @return
	 * @throws SequenceException
	 */
	public static Sequence newSequence(List<Vector> elements) throws SequenceException {
		checkElements(elements);
		return new Sequence(elements);
	}
	
	/**
	 * Copy the rhs into a new sequence
	 * @param rhs
	 * @return
	 * @throws SequenceException
	 */
	public static Sequence copySequence(Sequence rhs) throws SequenceException {
		return new Sequence(Collections.EMPTY_LIST).join(rhs);
	}
	
	/**
	 * Join the rhs sequence to the end of this sequence
	 * The resulting sequence is a new object
	 * @param rhs
	 * @return
	 * @throws SequenceException
	 */
	public Sequence join(Sequence rhs) throws SequenceException {
		List<Vector> allElements = new ArrayList<>(elements);
		for (Iterator<Vector> it = rhs.getIterator(); it.hasNext(); ) {
			allElements.add(it.next());
		}
		return Sequence.newSequence(allElements);
	}

	/**
	 * Return the length of the sequence ie the number of time-steps
	 * @return
	 */
	public int getLength() {
		return elements.size();
	}
	 
	 /**
	  * Get an iterator to the sequence
	  * which can be used independently of the internal iterator
	  * used by getNext()
	  * @return Iterator
	  */
	 public Iterator<Vector> getIterator() {
		 return elements.iterator();
	 }
	 
	
	@Override
	/**
	 * Return a string representation of the sequence.
	 */
	public String toString() {
		// for now just delegate to Object
		return super.toString();
	}
	
	@Override
	public boolean equals(Object right) {
		if (this == right)
			return true;
		
		if (!(right instanceof Sequence))
			return false;
		
		Sequence rhs = (Sequence)right;
		
		if (this.getLength() != rhs.getLength())
			return false;
		
		boolean result = true;
		Iterator<Vector> iterLhs = elements.iterator();
		Iterator<Vector> iterRhs = rhs.getIterator();
		while (iterLhs.hasNext()) {
			result = result && iterLhs.next().equals(iterRhs.next());
		}
		
		return result;
	}
	
	@Override
	public int hashCode() {
		int result = 17;
		Iterator<Vector> iter = elements.iterator();
		while (iter.hasNext()) {
			result = 31 * result + iter.next().hashCode();
		}
		return result;
	}

	public Vector get(int i) {
		return elements.get(i);
	}
 }
