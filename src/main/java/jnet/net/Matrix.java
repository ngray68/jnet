package jnet.net;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Matrix {

	private final Double[][] values;
	private final int rows;
	private final int cols;
	
	public Matrix(List<List<Double>> values) {
		this.rows = values.size();
		this.cols = values.get(0).size();
		this.values = new Double[rows][cols];
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				this.values[i][j] = values.get(i).get(j);
			}
		}	
	}
	
	public Matrix(Double[][] values) {
		assert (values.length > 0);
		assert (values[0].length > 0);
		this.rows = values.length;
		this.cols = values[0].length;
		this.values = new Double[rows][cols];
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				this.values[i][j] = values[i][j];
			}
		}	
	}
	
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.values = new Double[rows][cols];
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				this.values[i][j] = 0.0;
			}
		}	
		
	}
	
	public Matrix(int rows, int cols, Random random) {
		this.rows = rows;
		this.cols = cols;
		this.values = new Double[rows][cols];
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				this.values[i][j] = random.nextGaussian();
			}
		}	
		
	}
	
	public int getNumRows() {
		return rows;
	}
	
	public int getNumCols() {
		return cols;
	}
	
	public double getElement(int row, int col) {
		assert (row >= 0 && row < rows);
		assert (col >= 0 && col < cols);
		
		return values[row][col];
		
	}

	@Override
	public boolean equals(Object right) {
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

	public static Matrix add(Matrix left, Matrix right) {
		assert (left.getNumRows() == right.getNumRows());
		assert (left.getNumCols() == right.getNumCols());
		Double[][] sum = new Double[left.getNumRows()][left.getNumCols()];
		for (int i = 0; i < left.getNumRows(); ++i) {
			for (int j = 0; j < left.getNumCols(); ++j) {
				sum[i][j] = left.getElement(i, j) + right.getElement(i,j);
			}
		}
		
		return new Matrix(sum);
	}
	
	public static Vector multiply(Matrix M, Vector v) {
		assert (M.getNumCols() == v.getSize());
	
		ArrayList<Double> result = new ArrayList<Double>();
		for (int i = 0; i < M.getNumRows(); ++i) {
			double ithRow = 0;
			for (int j = 0; j < M.getNumCols(); ++j) {
				ithRow = ithRow + M.getElement(i, j) * v.getElement(j);
			}
			result.add(ithRow);
		}
		return new Vector(result);
	}
	
	public static Matrix multiply(double scalar, Matrix M) {
		Double[][] result = new Double[M.getNumRows()][M.getNumCols()];
		for (int i = 0; i < M.getNumRows(); ++i) {
			for (int j = 0; j < M.getNumCols(); ++j) {
				result[i][j] = M.getElement(i, j) * scalar;
			}
		}
		return new Matrix(result);
	}
	
	public static Matrix transpose(Matrix M) {
		List<List<Double>> transpose = new ArrayList<List<Double>>(M.getNumCols());
		for (int i = 0; i < M.getNumCols(); ++i) {
			transpose.add(i, new ArrayList<Double>(M.getNumRows()));
			for (int j = 0; j < M.getNumRows(); ++j) {
				transpose.get(i).add(j, M.getElement(j,i));
			}
		}
		return new Matrix(transpose);
	}
}
