package com.ngray.jnet.recurrent;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.ngray.jnet.algebra.Vector;

/**
 * Maps objects of type E to a unique one-hot Vector
 * The dimension of the Vector is the number of unique
 * objects of type E.
 * @author nigelgray
 *
 * @param <E>
 */
public final class Dictionary<E> {
	
	private final Map<E, Vector> lookup;
	private final Map<Vector, E> reverseLookup;
	private final E unknownInstance;
	
	private Dictionary(E unknownInstance) {
		lookup = new HashMap<>();
		reverseLookup = new HashMap<>();
		this.unknownInstance = unknownInstance;
	}
	
	private Dictionary(Map<E, Vector> definition, E unknownInstance) {
		lookup = new HashMap<>(definition);
		reverseLookup = new HashMap<>();
		lookup.forEach((e, vector) -> reverseLookup.put(vector, e));
		this.unknownInstance = unknownInstance;
	}
	
	/**
	 * Static factory method to return a new Dictionary
	 * @param definition
	 * @return
	 * @throws DictionaryException 
	 */
	public static <E> Dictionary<E> newDictionary(Map<E, Vector> definition, E unknownInstance) throws DictionaryException {
		// check that the dictionary is valid ie. each key maps to a unique vector
		// AND (TODO) check that the dictionary contains a special unknown instance mapping so we can capture
		// elements we don't recognize
		checkValidity(definition, unknownInstance);
		return new Dictionary<E>(definition, unknownInstance);
	}

	/**
	 * Static factory method to return a new empty dictionary
	 * @return
	 */
	public static <E> Dictionary<E> newDictionary(E unknownInstance) {
		return new Dictionary<>(unknownInstance);
	}
	
	/**
	 * Return a shallow copy of this dictionary
	 * @return
	 */
	public Dictionary<E> copy() {
		Dictionary<E> newDictionary = new Dictionary<>(unknownInstance);
		lookup.keySet().forEach((entry) -> {
			try {
				newDictionary.addEntry(entry, lookup.get(entry));
			} catch (DictionaryException e) {
				// we can never get here since we can't create a dictionary with duplicate entries
			}
		});
		return newDictionary;
	}
	
	public Set<E> getEntries() {
		return lookup.keySet();
	}
 	
	public Vector getVector(E instance) {
		if(lookup.containsKey(instance)) {
			return lookup.get(instance);
		}
		return lookup.get(unknownInstance);
	}

	public E getInstance(Vector v) {
		return reverseLookup.get(v);
	}
	
	public void addEntry(E e, Vector v) throws DictionaryException {
		if (lookup.containsKey(e) || reverseLookup.containsKey(v)) {
			throw new DictionaryException("Attempt to add duplicate key or vector to dictionary");
		}
		lookup.put(e, v);
		reverseLookup.put(v, e);
	}
	
	private static <E> void checkValidity(Map<E, Vector> definition, E unknownInstance) throws DictionaryException {
		if (!definition.containsKey(unknownInstance)) {
			throw new DictionaryException("The map contains no entry corresponding to the unknown instance");
		}
		Set<E> keys = definition.keySet();
		if (keys.size() != definition.size()) {
			throw new DictionaryException("There are multiple entries in the dictionary for a given key");
		}
		
		Set<Vector> vectors = definition.values().stream().collect(Collectors.toSet());
		if (vectors.size() != definition.size()) {
			throw new DictionaryException("Each unique key must map to a unique Vector in the dictianary");
		}
	}
}
