package com.ngray.jnet.recurrent.test;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.ngray.jnet.algebra.Vector;
import com.ngray.jnet.recurrent.Dictionary;
import com.ngray.jnet.recurrent.DictionaryException;
import com.ngray.jnet.recurrent.SequenceGenerator;

public final class Alphabet {
	
	public final static char[] SYMBOLS = {
			'~','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
			'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z',
			'!', '@', '£', '$', '%', '&', '*', '(', ')', ';', ':', '?', '-', ',', '.', ' ', '#', '€', '"', '{', '}', '[', ']',
			'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
	};
	
	public static int getSize() {
		return SYMBOLS.length;
	}
	
	public static Dictionary<Character> createDictionary() throws DictionaryException {
		
		Map<Character, Vector> map = new HashMap<>();
		for (int i = 0; i < getSize(); ++i) {
			Vector vector = new Vector(getSize(), 0.0);
			vector.setElement(i, 1.0);
			map.put(SYMBOLS[i], vector);
		}
		return Dictionary.newDictionary(map, '~');	
	}
	
	public static SequenceGenerator<Character> createSequenceGenerator() throws DictionaryException {
		return new SequenceGenerator<>(createDictionary());
	}
	
	public static String convert(List<Character> elements) {
		String s = "";
		for (Iterator<Character> iter = elements.iterator(); iter.hasNext(); ) {
			s += iter.next();
		}
		return s;
	}
	

}
