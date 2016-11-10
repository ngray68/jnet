package jnet.net.test;

import jnet.net.Matrix;
import jnet.net.Vector;
import junit.framework.TestCase;

public class TestMatrix extends TestCase {

	public void testGetNumRows() {
		//fail("Not yet implemented");
		Matrix matrix = new Matrix(4,5);
		assert (matrix.getNumRows() == 4);
	}

	public void testGetNumCols() {
		//fail("Not yet implemented");
		Matrix matrix = new Matrix(4,5);
		assert (matrix.getNumCols() == 5);
	}

	public void testGetElement() {
		//fail("Not yet implemented");
		Matrix matrix = new Matrix(new Double[][] { {5.0} });
		assert (matrix.getElement(0, 0) == 5.0);
	}

	public void testAdd() {
		//fail("Not yet implemented");
		Matrix left = new Matrix(new Double[][] { {5.0,6.0}, {4.0, 5.0} });
		Matrix right = new Matrix(new Double[][] { {2.0, 3.0}, {3.0, 2.0} });
		Matrix sum = new Matrix(new Double[][]{ {7.0, 9.0}, {7.0, 7.0} });
		assert (Matrix.add(left, right).equals(sum));
	}

	public void testTranspose() {
		//fail("Not yet implemented");
		Matrix matrix = new Matrix(new Double[][] {
			{1.0,2.0,3.0},
			{4.0,5.0,6.0}
		});
		
		Matrix transpose = new Matrix(new Double[][] {
			{1.0, 4.0},
			{2.0, 5.0},
			{3.0, 6.0}
		});
		
		assert (Matrix.transpose(matrix).equals(transpose));
	}

	public void testMultiply() {
		//fail("Not yet implemented");
		Matrix M = new Matrix(new Double[][] {
			{1.0,2.0,3.0},
			{4.0,5.0,6.0}
		});
		
		Vector v = new Vector(new Double[] {4.0, 5.0, 6.0});
		Vector product = new Vector(new Double[] {32.0, 77.0});		
		assert (Matrix.multiply(M, v).equals(product));
	}

}
