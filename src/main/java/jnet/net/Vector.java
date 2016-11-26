package jnet.net;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Vector {
	
	private final Double[] elements;
	private final int size;
	
	public Vector(List<Double> elements) {
		this.elements = new Double[elements.size()];
		int i = 0;
		for (Double element : elements) {
			this.elements[i] = element;
			++i;
		}
		this.size = elements.size();
	}
	
	public Vector(Double[] elements) {
		this.size = elements.length;
		this.elements = new Double[this.size];
		for (int i = 0; i < elements.length; ++i) {
			this.elements[i] = elements[i];
		}
	}
	
	public Vector(int size) {
		this.size = size;
		this.elements = new Double[size];
		for (int i = 0; i < size; ++i) {
			this.elements[i] = 0.0;
		}
	}
	
	public Vector(int size, Random random) {
		this.size = size;
		this.elements = new Double[size];
		for (int i = 0; i < size; ++i) {
			this.elements[i] = random.nextGaussian();
		}
	}
		
	public Vector(int size, double value) {
		this.size = size;
		this.elements = new Double[size];
		for (int i = 0; i < size; ++i) {
			this.elements[i] = value;
		}
	}

	public int getSize() {
		return size;
	}
	
	public double getElement(int i) {
		assert(i < getSize());
		return elements[i];
	}
	
	@Override
	public boolean equals(Object right) {
		if (this == right)
			return true;
		
		Vector rhs = (Vector)right;
		if (this.getSize() != rhs.getSize())
			return false;
		
		for (int i = 0; i < this.getSize(); ++i) {
			if (this.getElement(i) != rhs.getElement(i))
				return false;
		}
		return true;
	}
	
	@Override
	public String toString() {
		String str = "(";
		for (int i = 0; i < getSize(); ++i) {
			str += String.format("%f", elements[i]);
			if (i < getSize() - 1) {
				str += ",";
			}
			else {
				str += ")";
			}
		}
		return str;
	}
	
	public static Vector add(Vector left, Vector right) {
		assert (left.getSize() == right.getSize());
		int i = 0;
		
		int size = left.getSize();
		ArrayList<Double> result = new ArrayList<Double>();
		while (i < size) {
			result.add(i, left.getElement(i) + right.getElement(i));
			++i;
		}
		return new Vector(result);
	}

	
	public static Vector multiply(double scalarValue, Vector vector) {
		int i = 0;
		
		int size = vector.getSize();
		ArrayList<Double> result = new ArrayList<Double>();
		while (i < size) {
			result.add(i, vector.getElement(i) * scalarValue);
			++i;
		}
		return new Vector(result);
	}
	
	public static double dotProduct(Vector left, Vector right) {
		assert (left.getSize() == right.getSize());
		int i = 0;
		
		int size = left.getSize();
		double result = 0;
		while (i < size) {
			result = result + left.getElement(i) * right.getElement(i);
			++i;
		}
		return result;
	}
	
	public static Vector schurProduct(Vector left, Vector right) {
		assert (left.getSize() == right.getSize());
		int i = 0;
		
		int size = left.getSize();
		ArrayList<Double> result = new ArrayList<Double>();
		while (i < size) {
			result.add(i, left.getElement(i) * right.getElement(i));
			++i;
		}
		return new Vector(result);
	}
	
	public static Matrix dyadicProduct(Vector left, Vector right) {
		List<List<Double>> dyad = new ArrayList<List<Double>>(left.getSize());
		for (int i = 0; i < left.getSize(); ++i) {
			dyad.add(i, new ArrayList<Double>(right.getSize()));
			for (int j = 0; j < right.getSize(); ++j) {
				dyad.get(i).add(j, left.getElement(i) * right.getElement(j));
			}
		}
		return new Matrix(dyad);
	}
	
}
