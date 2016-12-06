package jnet.net;

import java.util.Random;

/**
 * Simple matrix implementation
 * Minimum needed for feed forward network
 * Possibly replace with a library implemention at some point
 * @author nigelgray
 *
 */
public class Matrix {
	
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
	 * Return a new Matrix element which is the sum of
	 * the two matrix parameters lhs and rhs. The supplied
	 * parameters must be the same size, or an AssertionError
	 * is thrown
	 * @param lhs
	 * @param rhs
	 * @return sum of lhs and rhs
	 */
	public static Matrix add(Matrix lhs, Matrix rhs)
	{
		assert (lhs.getNumRows() == rhs.getNumRows());
		assert (lhs.getNumCols() == rhs.getNumCols());
		Matrix result = new Matrix(lhs.getNumRows(), lhs.getNumCols());
		for (int i = 0; i < lhs.getNumRows(); ++i) {
			for (int j = 0; j < lhs.getNumCols(); ++j) {
				result.setElement(i, j, lhs.getElement(i, j) + rhs.getElement(i, j));
			}
		}
		return result;
	}
	
	/**
	 * Multiply the matrix M by the scalar quantity
	 * @param scalar
	 * @param M
	 * @return
	 */
	public static Matrix multiply(double scalar, Matrix M)
	{
		Matrix result = new Matrix(M.getNumRows(), M.getNumCols());
		for (int i = 0; i < M.getNumRows(); ++i) {
			for (int j = 0; j < M.getNumCols(); ++j) {
				result.setElement(i, j, M.getElement(i, j) * scalar);
			}
		}
		return result;
	}
	
	/**
	 * Multiply the vector V by the Matrix M. M must have the
	 * same number of cols as dimension of V.
	 * @param M
	 * @param v
	 * @return Vector which is the product of M and V
	 */
	public static Vector multiply(Matrix M, Vector v) 
	{
		assert (M.getNumCols() == v.getSize());
		
		Vector result = new Vector(M.getNumRows());
		for (int i = 0; i < M.getNumRows(); ++i) {		
			double sum = 0;
			for (int j = 0; j < M.getNumCols(); ++j) {
				sum = sum + M.getElement(i, j) * v.getElement(j);
			}
			result.setElement(i, sum);				
		}
		
		return result;
	}
	
	/**
	 * Return a new Matrix which is the transpose of M
	 * @param M
	 * @return
	 */
	public static Matrix transpose(Matrix M) 
	{
		Matrix result = new Matrix(M.getNumCols(), M.getNumRows());
		for (int i = 0; i < M.getNumCols(); ++i) {
			for (int j = 0; j < M.getNumRows(); ++j) {
				result.setElement(i, j, M.getElement(j, i));
			}
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
		Matrix rhs = (Matrix)right;
		if (this == right)
			return true;
		
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
	 * This matrix contains the sum on completion
	 * Matrices must be the same size or an AssertionError
	 * is thrown
	 * @param rhs
	 * @return
	 */
	public Matrix add(Matrix rhs)
	{
		assert (this.getNumRows() == rhs.getNumRows());
		assert (this.getNumCols() == rhs.getNumCols());
		for (int i = 0; i < this.getNumRows(); ++i) {
			for (int j = 0; j < this.getNumCols(); ++j) {
				this.setElement(i, j, this.getElement(i, j) + rhs.getElement(i, j));
			}
		}
		return this;
	}
}
