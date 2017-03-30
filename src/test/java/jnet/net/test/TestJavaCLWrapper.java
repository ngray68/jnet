package jnet.net.test;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ngray.jnet.algebra.JavaCLWrapper;
import com.ngray.jnet.algebra.Matrix;
import com.ngray.jnet.algebra.Vector;

public class TestJavaCLWrapper {

	@Test
	public void testMatrixMultiply() {
		// Test Multiplying a matrix by its transpose
		Matrix matrix = new Matrix(new double[][] {
			{1.0,2.0,3.0},
			{4.0,5.0,6.0}
		});
		
		Matrix transpose = new Matrix(new double[][] {
			{1.0, 4.0},
			{2.0, 5.0},
			{3.0, 6.0}
		});
		
		Matrix javaCLResult = JavaCLWrapper.multiply(matrix, transpose);
		Matrix jvmResult = matrix.multiply(transpose);
		assertTrue(javaCLResult.equals(jvmResult));
		
	}
	
	@Test
	public void testMatrixVectorMultiply() {
		// Test Multiplying a matrix by its transpose
		Matrix matrix = new Matrix(new double[][] {
			{1.0,2.0,3.0},
			{4.0,5.0,6.0}
		});
		
		Vector vector = new Vector(new double[] {1.0,2.0, 5.0});
		
		Vector javaCLResult = JavaCLWrapper.multiply(matrix, vector);
		Vector jvmResult = matrix.multiply(vector);
		assertTrue(javaCLResult.equals(jvmResult));
		
	}
	
	@Test
	public void performanceTestMatrixMultiply() {
		System.out.println("Testing performance of matrix/matrix multiplication");
		Matrix lhs = new Matrix(512, 512, 5.0);
		Matrix rhs = new Matrix(512, 512, 5.0);
		
		long start = System.currentTimeMillis();
		Matrix javaCLResult = JavaCLWrapper.multiply(lhs, rhs);
		long end = System.currentTimeMillis();
		System.out.println(end-start);
		
		start = System.currentTimeMillis();
		Matrix jvmResult = lhs.multiply(rhs);
		end = System.currentTimeMillis();
		System.out.println(end-start);
		assertTrue(javaCLResult.equals(jvmResult));
		
		
	}
	
	@Test
	public void performanceTestMatrixVectorMultiply() {
		System.out.println("Testing performance of matrix/vector multiplication");
		Matrix lhs = new Matrix(512, 512, 5.0);
		Vector rhs = new Vector(512, 5.0);
		
		long start = System.currentTimeMillis();
		Vector javaCLResult = JavaCLWrapper.multiply(lhs, rhs);
		long end = System.currentTimeMillis();
		System.out.println(end-start);
		
		start = System.currentTimeMillis();
		Vector jvmResult = lhs.multiply(rhs);
		end = System.currentTimeMillis();
		System.out.println(end-start);
		assertTrue(javaCLResult.equals(jvmResult));
		
		
	}

}
