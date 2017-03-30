package com.ngray.jnet.algebra;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Simple Vector implementation for neural nets
 * @author nigelgray
 *
 */
public final class Vector {
	
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
		
		if (!(right instanceof Vector)) 
			return false;
				
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
	
	@Override
	public int hashCode() {
		return Arrays.hashCode(elements);
	}
	
	/**
	 * Add rhs to this Vector. Result is a new vector
	 * Vectors must be the same size.
	 * @param right
	 * @return new vector which is the sum of this and right
	 */
	
	public Vector add(Vector right) {
		assert (this.getSize() == right.getSize());
		int i = 0;
		
		int size = this.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, this.getElement(i) + right.getElement(i));
			++i;
		}
		return result;
	}
	
	/**
	 * Subtract rhs from this Vector. Result is a new vector
	 * Vectors must be the same size.
	 * @param right
	 * @return new vector which is the sum of this and right
	 */
	public Vector subtract(Vector right) {
		assert (this.getSize() == right.getSize());
		int i = 0;
		
		int size = this.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, this.getElement(i) - right.getElement(i));
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
	public Vector multiply(double scalarValue) {
		int i = 0;
		
		int size = this.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, this.getElement(i) * scalarValue);
			++i;
		}
		return result;
	}
	
	/**
	 * Return the dotproduct (scalar) of the two vectors
	 * 
	 * @param right
	 * @return
	 */
	public double dotProduct(Vector right) {
		assert (this.getSize() == right.getSize());
		int i = 0;
		
		int size = this.getSize();
		double result = 0;
		while (i < size) {
			result = result + this.getElement(i) * right.getElement(i);
			++i;
		}
		return result;
	}
	
	/**
	 * Return the schur product of the two vectors
	 * 
	 * @param right
	 * @return
	 */
	public Vector schurProduct(Vector right) {
		assert (this.getSize() == right.getSize());
		int i = 0;
		
		int size = this.getSize();
		Vector result = new Vector(size);
		while (i < size) {
			result.setElement(i, this.getElement(i) * right.getElement(i));
			++i;
		}
		return result;
	}
	
	/**
	 * Return the dyadic product of the two vectors
	 *
	 * @param right
	 * @return
	 */
	public Matrix dyadicProduct(Vector right) {
		Matrix dyad = new Matrix(this.getSize(), right.getSize());
		for (int i = 0; i < this.getSize(); ++i) {
			for (int j = 0; j < right.getSize(); ++j) {
				dyad.setElement(i, j, this.getElement(i) * right.getElement(j));
			}
		}
		return dyad;
	}
	
	
	public float[] asFloatArray() {
		float[] floatArray = new float[elements.length];
		for (int i = 0; i < elements.length; ++i) {
			floatArray[i] = (float)elements[i];
		}
		return floatArray;
	}
	
	public static Vector fromFloatArray(float[] floatArray, int size) {
		Vector result = new Vector(size);
		for (int i = 0; i < size; ++i) {
				result.setElement(i, (double)floatArray[i]);
		}
		return result;
	}
}
