package jnet.net;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Simple Vector implementation for neural nets
 * @author nigelgray
 *
 */
public class Vector {
	
	/**
	 * The elements of the vector
	 */
	private final double[] elements;
	
	/**
	 * The number of elements in the vector (its dimension)
	 */
	private final int size;
	
	/**
	 * Construct a vector from the List of Doubles
	 * @param elements
	 */
	public Vector(List<Double> elements) 
	{
		this.elements = new double[elements.size()];
		int i = 0;
		for (Double element : elements) {
			this.elements[i] = element;
			++i;
		}
		this.size = elements.size();
	}
	
	/**
	 * Construct a vector from the array of doubles
	 * @param elements
	 */
	public Vector(double[] elements) 
	{
		this.size = elements.length;
		this.elements = new double[this.size];
		for (int i = 0; i < elements.length; ++i) {
			this.elements[i] = elements[i];
		}
	}
	
	/**
	 * Construct a zero-initialized vector of the specified size
	 * @param size
	 */
	public Vector(int size) {
		this.size = size;
		this.elements = new double[size];
		for (int i = 0; i < size; ++i) {
			this.elements[i] = 0.0;
		}
	}
	
	/**
	 * Construct a randomly initialized vector of the specified size
	 * Random values are between 0 and 1 with a Guassian distribution
	 * @param size
	 * @param random
	 */
	public Vector(int size, Random random) {
		this.size = size;
		this.elements = new double[size];
		for (int i = 0; i < size; ++i) {
			this.elements[i] = random.nextGaussian();
		}
	}
		
	/**
	 * Construct a vector of the specified size whose elements are all
	 * initialized to the specified value
	 * @param size
	 * @param value
	 */
	public Vector(int size, double value) {
		this.size = size;
		this.elements = new double[size];
		for (int i = 0; i < size; ++i) {
			this.elements[i] = value;
		}
	}

	/**
	 * Get the vector's size
	 * @return
	 */
	public int getSize() {
		return size;
	}
	
	/**
	 * Get the ith element of the vector
	 * i must be less than the vector's size
	 * @param i
	 * @return
	 */
	public double getElement(int i) {
		assert(i < getSize());
		return elements[i];
	}
	
	/**
	 * Set the ithe element to the specified value
	 * i must be less than the vector's size
	 * @param i
	 * @param value
	 */
	public void setElement(int i, double value)
	{
		assert(i < getSize());
		elements[i] = value;
	}
	
	@Override
	public boolean equals(Object right) 
	{
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
	public String toString() 
	{
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
	
	/**
	 * Add rhs to this Vector.This Vector will contain the sum
	 * on completion. Vectors must be the same size.
	 * @param right
	 * @return this
	 */
	public Vector add(Vector right) {
		assert (this.getSize() == right.getSize());
		int i = 0;
		
		int size = this.getSize();
		while (i < size) {
			elements[i] = this.getElement(i) + right.getElement(i);
			++i;
		}
		return this;
	}
	
	/**
	 * Return a new vector which is the sum of left and right.
	 * Left and right must be the same size
	 * @param left
	 * @param right
	 * @return
	 */
	public static Vector add(Vector left, Vector right) {
		assert (left.getSize() == right.getSize());
		int i = 0;
		
		int size = left.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, left.getElement(i) + right.getElement(i));
			++i;
		}
		return result;
	}

	/**
	 * Return a new vector by multiplying the vector V by the scalar value
	 * 
	 * @param scalarValue
	 * @param vector
	 * @return
	 */
	public static Vector multiply(double scalarValue, Vector vector) {
		int i = 0;
		
		int size = vector.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, vector.getElement(i) * scalarValue);
			++i;
		}
		return result;
	}
	
	/**
	 * Return the dotproduct (scalar) of the two vectors
	 * @param left
	 * @param right
	 * @return
	 */
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
	
	/**
	 * Return the schur product of the two vectors
	 * @param left
	 * @param right
	 * @return
	 */
	public static Vector schurProduct(Vector left, Vector right) {
		assert (left.getSize() == right.getSize());
		int i = 0;
		
		int size = left.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, left.getElement(i) * right.getElement(i));
			++i;
		}
		return result;
	}
	
	/**
	 * Return the dyadic product of the two vectors
	 * @param left
	 * @param right
	 * @return
	 */
	public static Matrix dyadicProduct(Vector left, Vector right) {
		Matrix dyad = new Matrix(left.getSize(), right.getSize());
		for (int i = 0; i < left.getSize(); ++i) {
			for (int j = 0; j < right.getSize(); ++j) {
				dyad.setElement(i, j, left.getElement(i) * right.getElement(j));
			}
		}
		return dyad;
	}
}
