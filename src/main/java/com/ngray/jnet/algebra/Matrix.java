package com.ngray.jnet.algebra;

import java.util.Arrays;
import java.util.Random;


/**
 * Simple matrix implementation
 * Minimum needed for feed forward network
 * Possibly replace with a library implemention at some point
 * @author nigelgray
 *
 */
public final class Matrix {
	
	/**
	 * The elements of the matrix
	 * Rows are offset by numCols
	 */
	private final double[] elements;
	
	/**
	 * The number of rows in the matrix
	 */
	private final int numRows;
	
	/**
	 * The number of columns in the matrix
	 */
	private final int numCols;
	
	/**
	 * Construct the identity matrix of size dim x dim
	 * @param rows
	 * @param cols
	 * @return
	 */
	public static Matrix identity(int dim) {
		Matrix ident = new Matrix(dim, dim);
		for (int i = 0; i < dim; ++i) {
			ident.setElement(i, i, 1.0);
		}
		return ident;
	}
	
	/**
	 * Multiply the matrix by the scalar quantity
	 * @param scalar
	 * @return new Matrix
	 */
	public Matrix multiply(double scalar)
	{
		Matrix result = new Matrix(getNumRows(), getNumCols());
		for (int i = 0; i < getNumRows(); ++i) {
			for (int j = 0; j < getNumCols(); ++j) {
				result.setElement(i, j, getElement(i, j) * scalar);
			}
		}
		return result;
	}
	
	/**
	 * Multiply the vector V by the Matrix. Matrix must have the
	 * same number of cols as dimension of V.
	 * @param v
	 * @return Vector which is the product of this and V
	 */
	public Vector multiply(Vector v) 
	{
		assert (getNumCols() == v.getSize());
		return JavaCLWrapper.multiply(this, v);
		/*
		Vector result = new Vector(getNumRows());
		for (int i = 0; i < getNumRows(); ++i) {		
			double sum = 0;
			for (int j = 0; j < getNumCols(); ++j) {
				sum = sum + getElement(i, j) * v.getElement(j);
			}
			result.setElement(i, sum);				
		}
		
		return result;
		*/
	}
	
	/**
	 * Multiply this matrix with the rhs
	 * @param rhs
	 * @return
	 */
	public Matrix multiply(Matrix rhs) 
	{
		assert (getNumCols() == rhs.getNumRows());	
		
		return JavaCLWrapper.multiply(this, rhs);
		/*
		Matrix result = new Matrix(getNumRows(), rhs.getNumCols());
		
		for (int i = 0; i < result.getNumRows(); ++i) {
			for (int j = 0; j < result.getNumCols(); ++j) {
				
				double value = 0.0;
				for (int k = 0; k < getNumCols(); ++k) {
					value += this.getElement(i, k) * rhs.getElement(k, j);
				}
				result.setElement(i, j, value);
			}
		}
		return result;
	*/
	}
	
	/**
	 * Multiply each element in the matrix by the corresponding element in the rhs
	 * @param rhs
	 * @return
	 */
	public Matrix multiplyElementWise(Matrix rhs) {
		assert (getNumRows() == rhs.getNumRows());	
		assert (getNumCols() == rhs.getNumCols());	
		Matrix result = new Matrix(getNumRows(), getNumCols());
		
		for (int i = 0; i < result.getNumRows(); ++i) {
			for (int j = 0; j < result.getNumCols(); ++j) {	
					result.setElement(i, j, getElement(i,j) * rhs.getElement(i, j));			
			}
		}
		return result;
	}
	
	/**
	 * Return a new Matrix which is the transpose of M
	 * @return
	 */
	public Matrix transpose() 
	{
		Matrix result = new Matrix(getNumCols(), getNumRows());
		for (int i = 0; i < getNumCols(); ++i) {
			for (int j = 0; j < getNumRows(); ++j) {
				result.setElement(i, j, getElement(j, i));
			}
		}
		return result;
	}
	
	/**
	 * If this matrix is square, return a vector of the elements on the diagonal
	 * @return
	 */
	public Vector diagonal() {
		assert (getNumRows() == getNumCols());
		Vector result = new Vector(getNumRows());
		for (int i = 0; i < result.getSize(); ++i) {
			result.setElement(i, getElement(i,i));
		}
		return result;
	}

	/**
	 * Construct a numRows x numCols matrix whose
	 * elements are all zero
	 * @param numRows
	 * @param numCols
	 */
	public Matrix(int numRows, int numCols)
	{
		elements = new double[numRows * numCols];
		this.numRows = numRows;
		this.numCols = numCols;
		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				elements[i * numCols + j] = 0.0;
			}
		}
	}
	
	/**
	 * Construct a numRows x numCols matrix whose
	 * elements are all initialized to the same value
	 * @param numRows
	 * @param numCols
	 * @param value
	 */
	public Matrix(int numRows, int numCols, double value)
	{
		elements = new double[numRows * numCols];
		this.numRows = numRows;
		this.numCols = numCols;
		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				elements[i * numCols + j] = value;
			}
		}
	}
	
	/**
	 * Construct a numRows x numCols matrix whose elements
	 * are randomly initialized with a Guassian distribution
	 * @param numRows
	 * @param numCols
	 * @param random
	 */
	public Matrix(int numRows, int numCols, Random random)
	{
		elements = new double[numRows * numCols];
		this.numRows = numRows;
		this.numCols = numCols;
		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				elements[i * numCols + j] = random.nextGaussian();
			}
		}
	}
	
	/**
	 * Construct a numRows x numCols matrix whose elements
	 * are randomly initialized with a Guassian distribution
	 * in the interval [-interval, interval]
	 * @param numRows
	 * @param numCols
	 * @param random
	 */
	public Matrix(int numRows, int numCols, Random random, double interval)
	{
		elements = new double[numRows * numCols];
		this.numRows = numRows;
		this.numCols = numCols;
		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				elements[i * numCols + j] = random.nextGaussian() * interval;
			}
		}
	}
	
	/**
	 * Construct a matrix which is initialized with the 2-dimensional
	 * array of doubles
	 * @param values
	 */
	public Matrix(double[][] values) 
	{
		assert (values.length > 0);
		assert (values[0].length > 0);
		numRows = values.length;
		numCols = values[0].length;
		elements = new double[numRows * numCols];
		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				elements[i * numCols + j] = values[i][j];
			}
		}
	}
	
	@Override
	public boolean equals(Object right) 
	{	
		if (this == right)
			return true;
		
		if (!(right instanceof Matrix)) 
			return false;
		
		Matrix rhs = (Matrix)right;
		
		if (this.getNumRows() != rhs.getNumRows() ||
			this.getNumCols() != rhs.getNumCols()) {
			return false;
		}
		
		for (int i = 0; i < this.getNumRows(); ++i) {
			for (int j = 0; j < this.getNumCols(); ++j) {
				if (this.getElement(i, j) != rhs.getElement(i, j))
					return false;
			}
		}
		return true;
	}
	
	@Override
	public int hashCode() {
		return Arrays.hashCode(elements);
	}
	
	/**
	 * Get the number of rows
	 * @return
	 */
	public int getNumRows()
	{
		return numRows;
	}
	
	/**
	 * Get the number of columns
	 * @return
	 */
	public int getNumCols()
	{
		return numCols;
	}
	
	/**
	 * Return the value of the element at (row,col)
	 * @param row
	 * @param col
	 * @return
	 */
	public double getElement(int row, int col)
	{
		return elements[row * getNumCols() + col];
	}
	
	/**
	 * Set the value of the element at (row, col)
	 * @param row
	 * @param col
	 * @param value
	 */
	public void setElement(int row, int col, double value)
	{
		elements[row * getNumCols() + col] = value;
	}
	
	/**
	 * Add the Matrix rhs to this Matrix
	 * The result is contained in a new Matrix returned
	 * Matrices must be the same size or an AssertionError
	 * is thrown
	 * @param rhs
	 * @return
	 */
	public Matrix add(Matrix rhs)
	{
		assert (this.getNumRows() == rhs.getNumRows());
		assert (this.getNumCols() == rhs.getNumCols());
		Matrix result = new Matrix(getNumRows(), getNumCols());
		for (int i = 0; i < this.getNumRows(); ++i) {
			for (int j = 0; j < this.getNumCols(); ++j) {
				result.setElement(i, j, this.getElement(i, j) + rhs.getElement(i, j));
			}
		}
		return result;
	}
	
	/**
	 * Subtract the Matrix rhs from this Matrix
	 * The result is contained in a new Matrix returned
	 * Matrices must be the same size or an AssertionError
	 * is thrown
	 * @param rhs
	 * @return
	 */
	public Matrix subtract(Matrix rhs) {
		assert (this.getNumRows() == rhs.getNumRows());
		assert (this.getNumCols() == rhs.getNumCols());
		Matrix result = new Matrix(getNumRows(), getNumCols());
		for (int i = 0; i < this.getNumRows(); ++i) {
			for (int j = 0; j < this.getNumCols(); ++j) {
				result.setElement(i, j, this.getElement(i, j) - rhs.getElement(i, j));
			}
		}
		return result;
	}
	
	@Override
	public String toString() {
		
		String result = "[\n";
		for (int i = 0; i < getNumRows(); ++i) {
			result += "  [ ";
			for (int j = 0; j < getNumCols(); ++j) {
				result += this.getElement(i,j);
				if (j < getNumCols() - 1) {
					result += ", ";
				}
			}
			result += " ]\n";
		}
		result += " ]";
		return result;
	}
	
	public float[] asFloatArray() {
		float[] floatArray = new float[elements.length];
		for (int i = 0; i < elements.length; ++i) {
			floatArray[i] = (float)elements[i];
		}
		return floatArray;
	}
	
	public static Matrix fromFloatArray(float[] floatArray, int rows, int cols) {
		Matrix result = new Matrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < rows; ++j) {
				result.setElement(i, j, (double)floatArray[i*cols + j]);
			}
		}
		return result;
	}
}
