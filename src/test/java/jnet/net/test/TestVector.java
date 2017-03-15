package jnet.net.test;

import java.util.ArrayList;

import com.ngray.jnet.algebra.Matrix;
import com.ngray.jnet.algebra.Vector;

import junit.framework.TestCase;

public class TestVector extends TestCase {
	
	protected void setUp() throws Exception {
		super.setUp();
	}

	protected void tearDown() throws Exception {
		super.tearDown();
	}

	public void testGetSize() {
		//fail("Not yet implemented");
		Vector v = new Vector(5);
		assert (v.getSize() == 5);
		
		v = new Vector(new ArrayList<Double>());
		assert (v.getSize() == 0);
	}

	public void testGetElement() {
		//fail("Not yet implemented");
		
		ArrayList<Double> contents = new ArrayList<Double>();
		contents.add(1.0);
		contents.add(2.0);
		contents.add(3.0);
		contents.add(4.0);
		contents.add(5.0);
		
		Vector v = new Vector(contents);
		assert (v.getElement(0) == 1);
		assert (v.getElement(1) == 2);
		assert (v.getElement(2) == 3);
		assert (v.getElement(3) == 4);
		assert (v.getElement(4) == 5);
	}
	
	
	public void testAdd() {
		//fail("Not yet implemented");
		ArrayList<Double> leftList = new ArrayList<Double>();
		leftList.add(1.0);
		leftList.add(2.0);
		ArrayList<Double> rightList = new ArrayList<Double>();
		rightList.add(1.0);
		rightList.add(2.0);
		ArrayList<Double> sumList = new ArrayList<Double>();
		sumList.add(2.0);
		sumList.add(4.0);
		Vector left = new Vector(leftList);
		Vector right = new Vector(rightList);
		Vector expectedSum = new Vector(sumList);
		Vector sum = left.add(right);
		assert (sum.equals(expectedSum));
		
		// test adding two vectors not the same size
		//leftList.add(5.0);
		//Vector newLeft = new Vector(leftList);
		//Vector.add(newLeft, left);
		
	}

	public void testMultiply() {
		//fail("Not yet implemented");
		ArrayList<Double> list = new ArrayList<Double>();
		list.add(1.0);
		list.add(2.0);
		Vector vector = new Vector(list);
		double scalar = 5.0;
		Vector result = vector.multiply(scalar);
		assert (result.getElement(0) == 5.0);
		assert (result.getElement(1) == 10.0);		
	}

	public void testDotProduct() {
		//fail("Not yet implemented");		
		Vector left = new Vector(new double[]{1.0, 2.0, 3.0});
		Vector right = new Vector(new double[]{2.0,3.0,4.0});
		double expectedResult = 20.0;
		assert (left.dotProduct(right) == expectedResult);
	}

	public void testSchurProduct() {
		//fail("Not yet implemented");
		Vector left = new Vector(new double[]{1.0, 2.0, 3.0});
		Vector right = new Vector(new double[]{2.0,3.0,4.0});
		Vector expectedResult = new Vector(new double[]{2.0, 6.0, 12.0});
		assert (left.schurProduct(right).equals(expectedResult));
	}

	public void testDyadicProduct() {
		//fail("Not yet implemented");
		Vector left = new Vector(new double[]{1.0, 2.0, 3.0});
		Vector right = new Vector(new double[]{2.0,3.0,4.0, 5.0});
		double[][] prod = new double[][] {
			{2.0, 3.0, 4.0, 5.0 },
			{4.0, 6.0, 8.0, 10.0},
			{6.0, 9.0, 12.0, 15.0}
		};
	
		Matrix result = new Matrix(prod);
		assert (left.dyadicProduct(right).equals(result));
	}
}