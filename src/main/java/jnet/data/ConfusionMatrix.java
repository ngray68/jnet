package jnet.data;

import jnet.net.Matrix;
import jnet.net.Vector;

/**
 * This class builds a confusion matrix from network output vectors
 * compared to the expected outputs for a given set of data instances
 * @author nigelgray
 *
 */
public class ConfusionMatrix {
	
	/**
	 * An NxN matrix 
	 * each row labels an expected output
	 * each column labels an actual output
	 */
	private Matrix matrix;
	
	/**
	 * Keep track of the number of data instances used to construct this matrix
	 */
	private int numInstances;
	
	/**
	 * Construct a confusion matrix and initialise with the given expected and actual
	 * and output
	 * @param expectedOutput
	 * @param networkOutput
	 */
	public ConfusionMatrix(Vector expectedOutput, Vector actualOutput)
	{
		assert (expectedOutput.getSize() == actualOutput.getSize());
		matrix = Vector.dyadicProduct(normalize(expectedOutput), normalize(actualOutput));
		numInstances = 1;
	}
	
	/**
	 * Update the confusion matrix with the specified expected and actual output
	 * Parameter sizes must be consistent with the confusion matrix ie.
	 * the size of the output vector must equal the number of rows or columns
	 * in the confusion matrix
	 * @param expectedOutput
	 * @param networkOutput
	 */
	public void addDataInstance(Vector expectedOutput, Vector actualOutput)  
	{
		assert (expectedOutput.getSize() == actualOutput.getSize());
		assert (matrix.getNumRows() == actualOutput.getSize()); 
		
		matrix = Matrix.add(matrix, Vector.dyadicProduct(normalize(expectedOutput), normalize(actualOutput)));
		++numInstances;
	}
	
	/**
	 * Return the accuracy - the number of true positives divided by the 
	 * total number of instances
	 * @return
	 */
	public double getAccuracy()
	{
		assert (numInstances != 0);
		double tp = 0.0;
		for (int i = 0; i < matrix.getNumRows(); ++i) {
			tp = tp + matrix.getElement(i, i);
		}
		return tp/numInstances;
	}
	
	/**
	 * Return a vector of precision for each class of output
	 * Precision is the number of true positives divided by
	 * the number of true positives and false positives
	 * @return
	 */
	public Vector getPrecision()
	{
		// precision is true positives/all observed positives for each class
		// ie. matrix element ii/sum over j of element ij
		Double[] precision = new Double[matrix.getNumRows()];
		for (int i = 0; i < matrix.getNumRows(); ++i) {
			Double tp = matrix.getElement(i, i);
			Double p = 0.0;
			for (int j = 0; j < matrix.getNumCols(); ++j) {
				p = p + matrix.getElement(i, j);
			}
			if (p == 0.0) {
				precision[i] = 0.0;
			} else {
				precision[i] = tp/p;
			}
		}
		return new Vector(precision);
	}
	
	/**
	 * Return a vector of recall for each class of output
	 * Recall is defined as the number of true positives
	 * divided by the sum of true positives and false negatives
	 * @return
	 */
	public Vector getRecall()
	{
		// recall is true positives/all actual positives for each class
		// ie. matrix element ii/sum over i of element ij
		Double[] recall = new Double[matrix.getNumRows()];
		for (int j = 0; j < matrix.getNumCols(); ++j) {
			Double tp = matrix.getElement(j, j);
			Double p = 0.0;
			for (int i = 0; i < matrix.getNumRows(); ++i) {
				p = p + matrix.getElement(i, j);
			}
			
			if (p == 0.0) {
				recall[j] = 0.0;
			} else {
				recall[j] = tp/p;
			}
		}
		return new Vector(recall);
	}
	
	/**
	 * Return the number of data instances used to construct this matrix
	 * @return the number of instances 
	 */
	public int getNumInstances()
	{
		return numInstances;
	}
	
	/**
	 * Helper method which normalizes a vector so that the sum of its elements
	 * is 1.0.
	 * @param v
	 * @return
	 */
	private Vector normalize(Vector v) 
	{
		assert (v != null);
		Vector unit = new Vector(v.getSize(), 1.0);
		double sum = Vector.dotProduct(unit, v);
		return Vector.multiply((1.0/sum), v);	
	}
	
}
